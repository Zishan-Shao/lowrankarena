import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.distributions as dist
from transformers import Trainer, TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
import numpy as np
import math
import random
from tqdm import tqdm
from pathlib import Path
import gc
import json
from datetime import datetime

from utils.datautils import prepare_train_loaders
from evaluate import evaluate_perplexity
from modules.stable_svd import stable_lowrank_SVD, SVDTransformLayer


def load_base_tokenizer(model_id: str, lower_id: str):
    """Load tokenizer robustly for old HF-converted LLaMA checkpoints like jeffwan/llama-7b-hf."""
    if "llama" in lower_id.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_id, legacy=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    if isinstance(tokenizer, bool):
        raise TypeError(
            f"Tokenizer load for {model_id} returned bool={tokenizer}. "
            "Use the repo's pinned env and do not pass use_fast to LLaMA tokenizer loading."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def current_gpu_mem_stats():
    if not torch.cuda.is_available():
        return {
            "device_index": None,
            "alloc_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_alloc_gb": 0.0,
            "peak_reserved_gb": 0.0,
        }

    device_idx = torch.cuda.current_device()
    gb = 1024 ** 3
    return {
        "device_index": int(device_idx),
        "alloc_gb": torch.cuda.memory_allocated(device_idx) / gb,
        "reserved_gb": torch.cuda.memory_reserved(device_idx) / gb,
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device_idx) / gb,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device_idx) / gb,
    }


class PerfCallback(TrainerCallback):
    def __init__(self, output_dir: Path, warmup_steps: int = 2, log_every: int = 1):
        self.output_dir = Path(output_dir)
        self.warmup_steps = warmup_steps
        self.log_every = log_every
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.step_t0 = None
        self.train_t0 = None
        self.step_times = []
        self.steps_path = self.output_dir / f"train_perf_rank{self.local_rank}.jsonl"
        self.summary_path = self.output_dir / f"train_perf_summary_rank{self.local_rank}.json"

    def _sync_and_reset(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())

    def on_train_begin(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_t0 = time.perf_counter()
        self._sync_and_reset()
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._sync_and_reset()
        self.step_t0 = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_t0 is None:
            return control

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_time_s = time.perf_counter() - self.step_t0
        mem = current_gpu_mem_stats()
        record = {
            "global_step": int(state.global_step),
            "epoch": None if state.epoch is None else float(state.epoch),
            "step_time_s": step_time_s,
            **mem,
        }
        with open(self.steps_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        if state.global_step > self.warmup_steps:
            self.step_times.append(step_time_s)

        if state.is_local_process_zero and self.log_every > 0 and state.global_step % self.log_every == 0:
            print(
                f"[perf] step={state.global_step} "
                f"time={step_time_s:.4f}s "
                f"peak_alloc={mem['peak_alloc_gb']:.4f}GB "
                f"reserved={mem['reserved_gb']:.4f}GB"
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_wall_time_s = None if self.train_t0 is None else (time.perf_counter() - self.train_t0)
        summary = {
            "rank": self.local_rank,
            "train_wall_time_s": train_wall_time_s,
            "num_profiled_steps_after_warmup": len(self.step_times),
            "mean_step_time_s": None,
            "p50_step_time_s": None,
            "p95_step_time_s": None,
            "final_mem": current_gpu_mem_stats(),
        }
        if self.step_times:
            summary["mean_step_time_s"] = float(np.mean(self.step_times))
            summary["p50_step_time_s"] = float(np.percentile(self.step_times, 50))
            summary["p95_step_time_s"] = float(np.percentile(self.step_times, 95))
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return control



def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # setting device
    print(f"Target Ratio: {args.target_ratio}")
    gpu_count = torch.cuda.device_count()
    print(f"Visible GPU count: {gpu_count}")
    NGPUS = gpu_count
    DEV_GPU = torch.device('cuda:0')
    DEV_CPU = torch.device('cpu')

    target_compression_ratio = args.target_ratio

    accelerator = Accelerator()

    SEQ_LEN = args.seq_len

    NSAMPLES_per_GPU_train = math.ceil(args.n_train_samples / NGPUS)
    NSAMPLES_per_GPU_val = math.ceil(args.n_eval_samples / NGPUS)

    SAVE = args.SAVE
    RECREATE = args.RECREATE
    DO_SAMPLE = args.DO_SAMPLE
    remapping = args.remapping

    NSAMPLES_train = args.n_train_samples
    NSAMPLES_val = args.n_eval_samples

    BETA = args.BETA

    # 训练设置, TA is TrainArgument
    TA_num_train_epochs = args.n_train_epochs
    TA_warmup_steps = args.warmup_steps
    TA_gradient_accumulation_steps = args.gradient_accumulation_steps
    save_epoch_num = args.save_epoch_num

    lambda_reg = args.lambda_reg

    # 优化器设置：
    scheduler_lr = args.scheduler_lr
    scheduler_step_size = math.ceil(NSAMPLES_per_GPU_train / TA_gradient_accumulation_steps)
    scheduler_gamma = args.scheduler_gamma
    scheduler_min_lr = args.scheduler_min_lr

    # load model
    model_load_dtype = torch.float16
    computeSVD_dtype = torch.float32

    model_id = args.model_id
    print("processing model: ", model_id.split('/')[-1])

    model_no_svd_layer_dic = {}
    lower_id = model_id.split('/')[-1]

    if "llama" in lower_id or "Llama" in lower_id:
        model_no_svd_layer_dic[lower_id] = ['lm_head']
    elif "opt" in lower_id:
        model_no_svd_layer_dic[lower_id] = ['project_out', 'project_in']

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=model_load_dtype)
    tokenizer = load_base_tokenizer(model_id, lower_id)

    # Set Path
    DATASET_NAME = args.training_dataset
    path_head_folder = Path(args.path_head_folder)
    path_head_folder_output = Path(args.path_head_folder_output)
    model_folder = path_head_folder / 'models' / lower_id
    data_cache_dir = path_head_folder / 'data_cache' / DATASET_NAME
    data_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir = path_head_folder / 'datasets' / DATASET_NAME
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = path_head_folder_output / 'training_output' / lower_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print("The output will be saved in: ", output_dir)

    # prepared dataset
    tokenized_traindata, tokenized_valdata = prepare_train_loaders(tokenizer, DATASET_NAME, data_cache_dir, dataset_cache_dir, args)

    # evaluate ppl of original model
    print("Start evaluating the original model's PPL.")
    val_input_ids = torch.cat([_["input_ids"].unsqueeze(0) for _ in tokenized_valdata], 0)
    model.to(DEV_GPU)
    orig_PPL = evaluate_perplexity(model, val_input_ids, NSAMPLES_val)
    model.to(DEV_CPU)
    print(f"Original Perplexity: {orig_PPL}")

    # transform model
    model_ori_weight_size = torch.tensor(0)
    for name, module in tqdm(model.named_modules(), desc="Add SVD attribute to modules"):
        if isinstance(module, nn.Linear) and all(x not in name for x in model_no_svd_layer_dic[lower_id]):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            attr_name = name.rsplit('.', 1)[-1]
            if parent_name != '':
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            RANK_RATIO = int(min(module.in_features, module.out_features) / SEQ_LEN)

            if remapping:
                gamma_init = (1 / RANK_RATIO) * target_compression_ratio * min(module.in_features, module.out_features)
            else:
                gamma_init = (1 / RANK_RATIO) * target_compression_ratio * module.in_features * module.out_features / (module.in_features + module.out_features)

            weight_size = torch.tensor(module.in_features * module.out_features)
            model_ori_weight_size += weight_size
            NewLayer = SVDTransformLayer(
                gamma=gamma_init,
                SEQ_LEN=SEQ_LEN,
                beta=BETA,
                input_size=module.in_features,
                output_size=module.out_features,
                weight_size=weight_size,
                weight=module.weight,
                bias=module.bias,
                name=name,
                device=model.device,
            )
            setattr(parent, attr_name, NewLayer)
            del module

    model.register_buffer('ori_weight_size', model_ori_weight_size)
    model.register_buffer('epoch_cnt', torch.tensor(0))
    model.register_buffer('BEST_loss', torch.tensor(float('inf'), dtype=computeSVD_dtype, device=model.device))
    gc.collect()
    print("transform done")

    # frozen other parameters
    now = datetime.now()
    formatted_time = now.strftime("%m%d-%H:%M:%S")
    if remapping:
        TA_tarined_model_output_dir = output_dir / f"Diff-Remapping-{target_compression_ratio}_{DATASET_NAME}_{SEQ_LEN}_{formatted_time}"
    else:
        TA_tarined_model_output_dir = output_dir / f"Diff-Noremapping-{target_compression_ratio}_{DATASET_NAME}_{SEQ_LEN}_{formatted_time}"

    for name, param in model.named_parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, SVDTransformLayer):
            module.gamma.requires_grad = True

    def calculate_compression_loss(model, target_compression_ratio, lambda_reg):
        size_new = torch.tensor(0.)

        for name, module in model.named_modules():
            if isinstance(module, SVDTransformLayer):
                RANK_RATIO = min(module.ori.in_features, module.ori.out_features) / SEQ_LEN
                if remapping:
                    size_now = max(module.ori.in_features, module.ori.out_features) * module.gamma * RANK_RATIO
                else:
                    size_now = module.ori.in_features * module.gamma * RANK_RATIO + module.ori.out_features * module.gamma * RANK_RATIO
                size_ori = module.ori_weight_size
                size_new = torch.where(size_now < size_ori, size_now, size_ori) + size_new

        compression_ratio = size_new / model.module.ori_weight_size

        compression_loss = abs(compression_ratio - torch.tensor(target_compression_ratio, device=compression_ratio.device))
        return lambda_reg * compression_loss, compression_ratio

    def Wrong_value_loss(model):
        penalty = torch.tensor(0., device=model.module.device)

        for name, module in model.named_modules():
            if isinstance(module, SVDTransformLayer):
                lower_penalty = torch.relu(-module.gamma) ** 2
                upper_penalty = torch.relu(module.gamma - torch.tensor(SEQ_LEN, device=module.gamma.device)) ** 2
                penalty += lower_penalty + upper_penalty

        return penalty

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    eval_perf_path = TA_tarined_model_output_dir / f"eval_perf_rank{local_rank}.jsonl"

    class SVDTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            neg_log_likelihood = loss
            ppl = torch.exp(neg_log_likelihood)
            loss = ppl

            reg_loss, compression_ratio = calculate_compression_loss(model, target_compression_ratio, lambda_reg)
            value_loss = Wrong_value_loss(model)

            total_loss = loss + reg_loss + value_loss

            cur_lr = self.optimizer.param_groups[0]['lr']

            model.module.epoch_cnt += 1
            if model.module.epoch_cnt % save_epoch_num == 0:
                k_dict = {}
                for name, module in self.model.named_modules():
                    if isinstance(module, SVDTransformLayer):
                        k_dict[name] = module.gamma.detach().item()
                k_dict['ppl'] = ppl.detach().tolist()
                k_dict['compression_ratio'] = compression_ratio.detach().tolist()
                k_dict['lr'] = cur_lr
                output_json_path = str(TA_tarined_model_output_dir / 'k_dict_{:05d}.json'.format(model.module.epoch_cnt))
                with open(output_json_path, 'w') as json_file:
                    json.dump(k_dict, json_file, indent=4)

                BEST_loss = model.module.BEST_loss
                CURR_loss = total_loss.mean().item()
                if CURR_loss < BEST_loss:
                    model.module.BEST_loss = torch.tensor(CURR_loss, device=model.module.BEST_loss.device)
                    k_dict["PPL_ORIG"] = orig_PPL
                    output_json_path = str(TA_tarined_model_output_dir / 'best_gamma.json')
                    with open(output_json_path, 'w') as json_file:
                        json.dump(k_dict, json_file, indent=4)
            return (total_loss, outputs) if return_outputs else total_loss

        def create_scheduler(self, num_training_steps, optimizer):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_step_size, eta_min=scheduler_min_lr)
            self.lr_scheduler = scheduler
            return self.lr_scheduler

        def create_optimizer(self):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=scheduler_lr)
            self.optimizer = optimizer
            return self.optimizer

        def evaluate(self, *args, **kwargs):
            if args_profile_train:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
                t0 = time.perf_counter()
                metrics = super().evaluate(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                record = {
                    "global_step": int(self.state.global_step),
                    "epoch": None if self.state.epoch is None else float(self.state.epoch),
                    "eval_time_s": time.perf_counter() - t0,
                    **current_gpu_mem_stats(),
                    "metrics": metrics,
                }
                with open(eval_perf_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                return metrics
            return super().evaluate(*args, **kwargs)

    args_profile_train = args.profile_train

    # training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=TA_tarined_model_output_dir,
        num_train_epochs=TA_num_train_epochs,
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=TA_warmup_steps,
        lr_scheduler_type="cosine",
        seed=args.seed,
        gradient_accumulation_steps=TA_gradient_accumulation_steps,
        save_strategy="no",
        save_steps=1000,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    callbacks = []
    if args.profile_train:
        callbacks.append(
            PerfCallback(
                TA_tarined_model_output_dir,
                warmup_steps=args.profile_warmup_steps,
                log_every=args.profile_log_every,
            )
        )

    trainer = SVDTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_traindata,
        eval_dataset=tokenized_valdata,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, trainer.get_train_dataloader(), trainer.get_eval_dataloader()
    )
    trainer.train = accelerator.prepare(trainer.train)

    # save setting
    TA_tarined_model_output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = TA_tarined_model_output_dir / "para_config.json"

    data = {
        "model_id": model_id,
        "DATASET_NAME": DATASET_NAME,
        "target_compression_ratio": target_compression_ratio,
        "BETA": BETA,
        "dataset_processing": {
            "SEED": args.seed,
            "SEQ_LEN": SEQ_LEN,
            "NSAMPLES_train": NSAMPLES_train,
            "NSAMPLES_val": NSAMPLES_val,
        },
        "training_settings": {
            "TA_num_train_epochs": TA_num_train_epochs,
            "TA_warmup_steps": TA_warmup_steps,
            "TA_gradient_accumulation_steps": TA_gradient_accumulation_steps,
        },
        "loss_function_settings": {
            "lambda_reg": lambda_reg,
        },
        "optimizer_settings": {
            "scheduler_lr": scheduler_lr,
            "scheduler_step_size": scheduler_step_size,
            "scheduler_gamma": scheduler_gamma,
        },
        "dtype_settings": {
            "model_load_dtype": "f16",
            "computeSVD_dtype": "f32",
        },
        "no_svd_layer": model_no_svd_layer_dic[lower_id],
        "profiling": {
            "enabled": args.profile_train,
            "profile_log_every": args.profile_log_every,
            "profile_warmup_steps": args.profile_warmup_steps,
        },
    }
    with open(output_json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print("training start, target_compression_ratio is ", target_compression_ratio, " , NGPUS is ", NGPUS)
    trainer.train()

    accelerator.wait_for_everyone()

    # evaluate activation-truncated model
    val_input_ids = torch.cat([_["input_ids"].unsqueeze(0) for _ in tokenized_valdata], 0)
    DASVD_PPL = evaluate_perplexity(model, val_input_ids, NSAMPLES_val)
    model = accelerator.unwrap_model(model)
    torch.cuda.empty_cache()
    print(f"Trained Perplexity: {DASVD_PPL}")
    k_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, SVDTransformLayer):
            k_dict[name] = module.gamma.detach().item()
    k_dict["PPL"] = {
        'orig_PPL': orig_PPL,
        'DASVD_PPL': DASVD_PPL,
    }
    output_json_path = str(TA_tarined_model_output_dir / 'final_gamma.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(k_dict, json_file, indent=4)

    print("Done, all thing you need is saved in ", TA_tarined_model_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Settings related to the model.
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID",
    )

    parser.add_argument(
        "--remapping",
        action="store_true",
        default=False,
        help="whether to use remaping to save model",
    )

    parser.add_argument(
        "--target_ratio",
        type=float,
        required=True,
        help="target param ratio",
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=2048,
        help='sequence length of model'
    )

    # Settings related to the Path.
    parser.add_argument(
        '--path_head_folder',
        type=str,
        default='./',
        help='path of the model and dataset'
    )

    parser.add_argument(
        '--path_head_folder_output',
        type=str,
        default='./results',
        help='path of the output result'
    )

    # Settings related to the device.
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )

    # Settings related to the datasets
    parser.add_argument(
        "--n_train_samples",
        type=int,
        default=256,
        help="number of samples used for finetuning",
    )

    parser.add_argument(
        "--n_eval_samples",
        type=int,
        default=256,
        help="number of samples used for evaluation",
    )

    parser.add_argument(
        "--training_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="finetuning dataset",
    )

    parser.add_argument(
        "--SAVE",
        action="store_true",
        default=False,
        help="whether to save the generated dataset",
    )

    parser.add_argument(
        "--RECREATE",
        action="store_true",
        default=True,
        help="whether to regenerate the dataset",
    )

    parser.add_argument(
        "--DO_SAMPLE",
        action="store_true",
        default=False,
        help="whether to obtain the data set by sampling",
    )

    # Settings related to the finetuning
    parser.add_argument(
        "--n_train_epochs",
        type=int,
        default=20,
        help="number of epoches of finetuning",
    )

    parser.add_argument(
        "--scheduler_lr",
        type=float,
        default=5,
        help="scheduler lr of finetuning",
    )

    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.9,
        help="scheduler gamma of finetuning",
    )

    parser.add_argument(
        "--scheduler_min_lr",
        type=float,
        default=4,
        help="scheduler min lr of finetuning",
    )

    parser.add_argument(
        "--BETA",
        type=int,
        default=10,
        help="beta of tanh in the finetuning",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="number of epoches of warmup",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="number of epoches of gradient accumulation",
    )

    parser.add_argument(
        "--save_epoch_num",
        type=int,
        default=8,
        help="number of epoches of save models",
    )

    parser.add_argument(
        "--lambda_reg",
        type=int,
        default=200,
        help="tuning parameters for model performance and size in training loss",
    )

    parser.add_argument(
        "--profile_train",
        action="store_true",
        default=False,
        help="log per-step training/eval wall time and CUDA memory into json/jsonl files",
    )

    parser.add_argument(
        "--profile_log_every",
        type=int,
        default=1,
        help="print one training perf line every N optimizer steps",
    )

    parser.add_argument(
        "--profile_warmup_steps",
        type=int,
        default=2,
        help="ignore the first N steps in aggregate timing summary",
    )

    args = parser.parse_args()
    main(args)
