#!/usr/bin/env python3
"""
Fine-tune a saved compressed model checkpoint.

Usage:
    python eval_encoder/finetune_from_checkpoint.py \
        --checkpoint eval_encoder/models/fwsvd_r300_flashsvd \
        --learning_rate 2e-5 \
        --num_epochs 3 \
        --batch_size 32
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm


TASK_CONFIG = {
    "sst2": {
        "num_labels": 2,
        "train_split": "train",
        "val_split": "validation",
        "sentence_keys": ("sentence",),
        "metric": "accuracy",
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune from checkpoint")

    # Model configuration
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved compressed model")
    parser.add_argument("--task", default="sst2", choices=["sst2"],
                        help="Task to fine-tune on")

    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Fine-tuning strategy
    parser.add_argument("--finetune_mode", default="full",
                        choices=["classifier_only", "full"],
                        help="What to fine-tune")

    # Data configuration
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=100)

    # Output configuration
    parser.add_argument("--output_dir", default="eval_encoder/finetuned_models")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(task, tokenizer, seq_len, batch_size):
    """Prepare train and validation dataloaders."""
    cfg = TASK_CONFIG[task]

    train_dataset = load_dataset("glue", task, split=cfg["train_split"])
    val_dataset = load_dataset("glue", task, split=cfg["val_split"])

    def tokenize(examples):
        keys = cfg["sentence_keys"]
        if len(keys) == 1:
            return tokenizer(
                examples[keys[0]],
                padding="max_length",
                truncation=True,
                max_length=seq_len
            )
        else:
            return tokenizer(
                examples[keys[0]], examples[keys[1]],
                padding="max_length",
                truncation=True,
                max_length=seq_len
            )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in train_dataset.column_names if c != "label"]
    )
    val_dataset = val_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in val_dataset.column_names if c != "label"]
    )

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch]),
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    from evaluate import load as load_metric

    metric = load_metric("accuracy")
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)
            metric.add_batch(
                predictions=preds.cpu(),
                references=batch["labels"].cpu()
            )

            total_loss += loss.item()
            num_batches += 1

    accuracy = metric.compute()["accuracy"]
    avg_loss = total_loss / max(num_batches, 1)

    return accuracy, avg_loss


def train(args):
    """Main training function."""
    set_seed(args.seed)

    # Load model and compression info
    checkpoint_path = Path(args.checkpoint)
    print(f"\n[load] Loading compressed model from {checkpoint_path}")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Load compression info
    info_file = checkpoint_path / "compression_info.json"
    if info_file.exists():
        with open(info_file) as f:
            comp_info = json.load(f)
        print(f"[info] Method: {comp_info['method']}")
        if comp_info['method'] == 'adasvd':
            print(f"[info] Budget: {comp_info['budget']}")
        else:
            print(f"[info] Rank: {comp_info['rank']}")
        print(f"[info] Backend: {comp_info['backend']}")
        print(f"[info] Accuracy before fine-tuning: {comp_info['accuracy_before_finetune']:.4f}")
    else:
        comp_info = None
        print("[warn] No compression_info.json found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare data
    train_loader, val_loader = prepare_data(
        args.task, tokenizer, args.seq_len, args.batch_size
    )

    # Configure fine-tuning mode
    if args.finetune_mode == "classifier_only":
        print("[finetune] Mode: Classifier only (freezing encoder)")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    else:
        print("[finetune] Mode: Full model")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[params] Trainable: {trainable_params/1e6:.1f}M / {total_params/1e6:.1f}M ({trainable_params/total_params*100:.1f}%)")

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Evaluate before training
    print("\n[eval] Evaluating before training...")
    val_acc_before, val_loss_before = evaluate(model, val_loader, device)
    print(f"[eval] Before: Accuracy={val_acc_before:.4f}, Loss={val_loss_before:.4f}")

    # Training loop
    print(f"\n[train] Starting training for {args.num_epochs} epochs...")
    print(f"[train] Total steps: {total_steps}")

    best_accuracy = val_acc_before
    best_model_state = None

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % args.eval_steps == 0:
                val_acc, val_loss = evaluate(model, val_loader, device)
                print(f"\n[eval] Step {global_step}: Accuracy={val_acc:.4f}, Loss={val_loss:.4f}")

                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    print(f"[eval] New best: {best_accuracy:.4f}")

                model.train()

        # End of epoch evaluation
        avg_train_loss = epoch_loss / len(train_loader)
        val_acc, val_loss = evaluate(model, val_loader, device)

        print(f"\n[epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
        print(f"[epoch {epoch+1}] Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[epoch {epoch+1}] New best: {best_accuracy:.4f}")

    # Load best model
    if best_model_state is not None:
        print("\n[final] Loading best model...")
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    final_acc, final_loss = evaluate(model, val_loader, device)

    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Before training: {val_acc_before:.4f}")
    print(f"After training:  {final_acc:.4f}")
    print(f"Best accuracy:   {best_accuracy:.4f}")
    print(f"Improvement:     +{(best_accuracy - val_acc_before)*100:.2f}%")
    print("="*60)

    # Save fine-tuned model
    save_model(model, tokenizer, args, best_accuracy, comp_info)

    return best_accuracy


def save_model(model, tokenizer, args, accuracy, comp_info):
    """Save fine-tuned model."""
    checkpoint_name = Path(args.checkpoint).name
    output_dir = Path(args.output_dir) / f"{checkpoint_name}_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[save] Saving fine-tuned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training info
    info = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "finetune_mode": args.finetune_mode,
        "final_accuracy": float(accuracy),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if comp_info:
        info["compression"] = comp_info

    with open(output_dir / "finetune_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"[save] Model saved to {output_dir}")


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("Fine-tuning Compressed Model")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode: {args.finetune_mode}")
    print("="*60 + "\n")

    best_accuracy = train(args)

    print("\n✅ Fine-tuning completed!")
    print(f"Best accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
