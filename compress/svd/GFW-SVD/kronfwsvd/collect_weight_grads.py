import argparse
import os
import re
import torch
import random
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from safetensors.torch import save_file
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from accelerate.accelerator import AcceleratorState
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time


def print_once(*args, **kwargs):
    if AcceleratorState.is_main_process:
        print(*args, **kwargs)


class NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self, param):
        dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        super().__init__([{"params": [dummy_param], "lr": 0.0}], {})

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class GradientTrainer(Trainer):
    def __init__(self, *args, layer_pattern=None, step_counter=None, save_queue=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_pattern = layer_pattern
        self.step_counter = step_counter
        self.save_queue = save_queue
        self._grad_buffer = {}
        self._current_acc_step = 0
        self._register_hooks()

    def _register_hooks(self):
        """LowRankArena extension: collect true weight gradients.

        Upstream maxim collects module grad_output. GFWSVD Algorithm 1
        requires G_i = dL/dW with the same shape as the linear weight.
        """
        matched = 0

        for name, param in self.model.named_parameters():
            if not name.endswith(".weight"):
                continue

            module_name = name[:-len(".weight")]

            if re.search(self.layer_pattern, module_name):
                if not param.requires_grad:
                    continue

                param.register_hook(self._make_hook(module_name))
                matched += 1
                print_once(
                    f"[DEBUG] Registered weight-gradient hook for: {name}"
                )

        if matched == 0:
            print_once(
                f"[WARNING] No weight parameters matched pattern "
                f"'{self.layer_pattern}'"
            )

    def _make_hook(self, name):
        def hook(grad):
            if grad is None:
                return

            grad = grad.detach()

            if name in self._grad_buffer:
                self._grad_buffer[name].add_(grad)
            else:
                self._grad_buffer[name] = grad.clone()

            # Do not modify the gradient used by autograd.
            return grad

        return hook

    def training_step(self, model, inputs, num_items):
        # Call superclass method to perform forward/backward
        loss = super().training_step(model, inputs, num_items)

        # Increment our step counter
        self._current_acc_step += 1
        print_once(
            f"[DEBUG] Step {self.state.global_step}, acc_step {self._current_acc_step}/{self.args.gradient_accumulation_steps}"
        )

        # Check if we've reached the end of gradient accumulation
        if self._current_acc_step >= self.args.gradient_accumulation_steps:
            print_once(f"[INFO] Saving gradients for batch {self.step_counter[0]}")

            # Process and save accumulated gradients
            # accelerator.backward() already divides the loss by
            # gradient_accumulation_steps. Summing hook gradients across
            # microsteps therefore already gives the accumulated average.
            grad_dict = {
                name: tensor.clone()
                for name, tensor in self._grad_buffer.items()
            }

            # Send for saving
            self.save_queue.put((self.step_counter[0], grad_dict))

            # Reset buffer and counter
            self._grad_buffer.clear()
            self._current_acc_step = 0
            self.step_counter[0] += 1

        return loss

    def create_optimizer(self):
        print_once("[INFO] Using dummy optimizer (no updates)")
        self.optimizer = NoOpOptimizer(next(self.model.parameters()))
        return self.optimizer


def save_gradient_worker(save_queue, output_path, flush_interval=5):
    """Worker thread function to save gradients to disk"""
    print_once("[INFO] Save worker started")
    last_flush_time = time.time()
    pending_saves = []
    save_count = 0

    while True:
        try:
            # Try to get an item with timeout to allow periodic flushing
            try:
                item = save_queue.get(timeout=1.0)
                if item is None:  # Poison pill
                    break
            except queue.Empty:
                # No new items, check if we should flush pending saves
                if pending_saves and time.time() - last_flush_time > flush_interval:
                    for batch_idx, grads in pending_saves:
                        _save_gradient_batch(batch_idx, grads, output_path)
                    pending_saves.clear()
                    last_flush_time = time.time()
                continue

            # Process the gradient save
            batch_idx, grads = item
            pending_saves.append((batch_idx, grads))

            # If we have enough pending saves or it's been too long, flush them
            if len(pending_saves) >= 3 or time.time() - last_flush_time > flush_interval:
                for b_idx, g in pending_saves:
                    _save_gradient_batch(b_idx, g, output_path)
                    save_count += 1
                pending_saves.clear()
                last_flush_time = time.time()
                print_once(f"[INFO] Saved {save_count} gradient batches so far")

            save_queue.task_done()

        except Exception as e:
            print_once(f"[ERROR] Exception in save worker: {e}")
            import traceback

            traceback.print_exc()

    # Final flush of any remaining items
    for batch_idx, grads in pending_saves:
        _save_gradient_batch(batch_idx, grads, output_path)
        save_count += 1

    print_once(f"[INFO] Save worker finished. Total saved batches: {save_count}")


def _save_gradient_batch(batch_idx, grads, output_path):
    """Helper function to save a single batch of gradients"""
    try:
        # Create a new dict for CPU tensors
        grads_cpu = {}

        # Process each gradient tensor
        for k, v in grads.items():
            # Move to CPU and convert to bfloat16
            cpu_tensor = v.detach().cpu().to(dtype=torch.bfloat16)
            grads_cpu[k] = cpu_tensor

        # Save to disk
        grad_file = os.path.join(output_path, f"grads_batch_{batch_idx:06}.safetensors")
        save_file(grads_cpu, grad_file)
        print_once(f"[DEBUG] Saved gradients to {grad_file}")

        # Clean up
        del grads_cpu

    except Exception as e:
        print_once(f"[ERROR] Failed to save gradient batch {batch_idx}: {e}")
        import traceback

        traceback.print_exc()


def collect_grads(
    output_path,
    model_name,
    dataset_name,
    dataset_config_name,
    dataset_text_field,
    batch_size,
    gradient_accumulation_steps,
    layer_pattern,
    max_steps,
    max_seq_length,
    seed,
    use_bf16,
    use_qlora,
    subsample_size,
    collect_weights,
    gradient_checkpointing,
    num_save_workers=2,
    compression_level=1,
):
    os.makedirs(output_path, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print_once(f"[INFO] Starting gradient collection with:")
    print_once(f"       - Model: {model_name}")
    print_once(f"       - Batch size: {batch_size}")
    print_once(f"       - Gradient accumulation steps: {gradient_accumulation_steps}")
    print_once(f"       - Layer pattern: {layer_pattern}")
    print_once(f"       - Output path: {output_path}")

    #tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if "gemma" not in model_name else "eager",
    )

    if gradient_checkpointing:
        if hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = True
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            print_once("[INFO] Enabling gradient checkpointing with non-reentrant mode")
            model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    if use_qlora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, lora_dropout=0.1, bias="none")
        model = get_peft_model(model, lora_config)

    for param in model.parameters():
        param.requires_grad = False

    unfrozen_count = 0
    for name, param in model.named_parameters():
        if re.search(layer_pattern, name):
            param.requires_grad = True
            unfrozen_count += 1
            print_once(f"[INFO] Unfrozen: {name}")

    if unfrozen_count == 0:
        print_once(f"[WARNING] No parameters matched the pattern '{layer_pattern}'. Check your regex.")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_once(f"[INFO] Total params: {total:,} | Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")

    if trainable == 0:
        print_once("[ERROR] No trainable parameters found. Exiting.")
        return

    model.train()

    subsample_path = os.path.join(dataset_name, "sample", str(subsample_size))
    if os.path.exists(subsample_path):
        print_once(f"[INFO] Loading cached subsampled dataset from: {dataset_name}/{subsample_size}")
        ds = load_from_disk(subsample_path)
    else:
        print_once(f"[INFO] Loading dataset: {dataset_name}")
        full_ds = load_dataset(dataset_name, dataset_config_name, split="train")

        if subsample_size:
            print_once(f"[INFO] Subsampling {subsample_size} examples...")
            ds = full_ds.shuffle(seed=seed).select(range(subsample_size))
        else:
            ds = full_ds
        print_once(f"[INFO] Saving subsampled dataset to {subsample_path}")
        os.makedirs(os.path.dirname(subsample_path), exist_ok=True)
        ds.save_to_disk(subsample_path)

    def tokenize_function(example):
        return tokenizer(
            example[dataset_text_field],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

    print_once("[INFO] Tokenizing in-memory...")
    tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    tokenized.set_format(type="torch")
    print_once("[INFO] Tokenization complete")

    # Stats
    lengths = [len(x["input_ids"]) for x in tokenized]
    total_tokens = sum(lengths)
    print_once(f"[INFO] Tokenization stats:")
    print_once(f"       Total examples: {len(tokenized)}")
    print_once(f"       Total tokens: {total_tokens:,}")
    print_once(f"       Avg length: {np.mean(lengths):.2f}")
    print_once(f"       Median length: {np.median(lengths):.2f}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Configure save queue with increased capacity
    save_queue = queue.Queue(maxsize=max(32, num_save_workers * 4))
    step_counter = [0]

    # Start save workers
    save_threads = []
    for _ in range(num_save_workers):
        save_thread = threading.Thread(target=save_gradient_worker, args=(save_queue, output_path), daemon=True)
        save_thread.start()
        save_threads.append(save_thread)
    print_once(f"[INFO] Started {num_save_workers} save worker threads")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,
        max_steps=max_steps if max_steps is not None else -1,
        logging_steps=10,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=[],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        save_strategy="no",
        save_steps=0,
        save_total_limit=0,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=0.0,  # Disable gradient clipping
    )

    if gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    trainer = GradientTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        layer_pattern=layer_pattern,
        step_counter=step_counter,
        save_queue=save_queue,
    )

    print_once("[INFO] Starting training loop")
    trainer.train()
    print_once("[INFO] Training finished. Waiting for gradient save queue to empty...")

    # Wait for the queue to be processed
    save_queue.join()

    # Now send termination signals to workers
    print_once("[INFO] Shutting down save workers...")
    for _ in range(num_save_workers):
        save_queue.put(None)  # Poison pill for each worker

    # Wait for all save threads to finish
    for thread in save_threads:
        thread.join()

    print_once(f"[INFO] All gradients saved. Total batches: {step_counter[0]}")


def main():
    parser = argparse.ArgumentParser(description="Collect gradients from a model without updating weights")
    parser.add_argument("--path_to", required=True, help="Output directory for collected gradients")
    parser.add_argument("--model_name", default="google/gemma-3b-it", help="Model to collect gradients from")
    parser.add_argument("--dataset_name", default="roneneldan/TinyStories", help="Dataset to use")
    parser.add_argument("--dataset_config_name", default=None, help="Dataset configuration")
    parser.add_argument("--dataset_text_field", default="text", help="Text field name in dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--layer_pattern", type=str, required=True, help="Regex pattern to match layer/module names")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Max training steps. Default: run through entire dataset."
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="Max sequence length for tokenization. Default: 2048."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 precision")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA for quantization")
    parser.add_argument("--subsample_size", type=int, default=None, help="Subsample dataset to this size")
    parser.add_argument("--collect_weights", action="store_true", default=False, help="Also collect weights")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing")
    parser.add_argument("--num_save_workers", type=int, default=2, help="Number of parallel save workers")
    parser.add_argument(
        "--compression_level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Compression level for saved gradients (0=none, 1=bfloat16, 2=int8)",
    )

    args = parser.parse_args()

    collect_grads(
        output_path=args.path_to,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_text_field=args.dataset_text_field,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        layer_pattern=args.layer_pattern,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        use_bf16=args.bf16,
        use_qlora=args.use_qlora,
        subsample_size=args.subsample_size,
        collect_weights=args.collect_weights,
        gradient_checkpointing=args.gradient_checkpointing,
        num_save_workers=args.num_save_workers,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    main()
