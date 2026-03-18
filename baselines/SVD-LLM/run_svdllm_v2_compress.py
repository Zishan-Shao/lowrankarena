"""
SVD-LLM V2 compression runner.
Calls whitening_hetero() from SVDLLM_v2.py and optionally whitening_local_update() for step-2 style.
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from SVDLLM_v2 import whitening_hetero
from SVDLLM import whitening_local_update
from utils.data_utils import get_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--profiling_mat_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--attn_ratio", type=float, default=None)
    parser.add_argument("--mlp_ratio", type=float, default=None)
    parser.add_argument("--svd_method", type=str, default="full", choices=["full", "randomized"])
    parser.add_argument("--local_update", action="store_true", help="apply whitening_local_update after whitening_hetero")
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--updating_nsamples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model.split("/")[-1].lower()

    print(f"Loading model: {args.model}")
    hf_kwargs = {"trust_remote_code": True}
    if args.hf_token:
        hf_kwargs["token"] = args.hf_token
    if args.local_update:
        # step 2 requires float32
        hf_kwargs["torch_dtype"] = torch.float32
    else:
        hf_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, **hf_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                              **({"token": args.hf_token} if args.hf_token else {}))
    model = model.to(dev)
    model.eval()
    model.seqlen = args.model_seq_len

    print(f"Loading profiling_mat: {args.profiling_mat_path}")
    profiling_mat = torch.load(args.profiling_mat_path, weights_only=False)
    # whitening_hetero doesn't cast to float32 internally; linalg.inv requires it
    profiling_mat = {
        i: {k: v.float() for k, v in layer.items()}
        for i, layer in profiling_mat.items()
    }

    print(f"Compressing with whitening_hetero (ratio={args.ratio}) ...")
    whitening_hetero(
        model_name=model_name,
        model=model,
        profiling_mat=profiling_mat,
        ratio=args.ratio,
        dev=dev,
        attn_ratio=args.attn_ratio,
        mlp_ratio=args.mlp_ratio,
        svd_method=args.svd_method,
    )

    os.makedirs(args.save_path, exist_ok=True)
    keep = 1 - args.ratio
    model_prefix = args.model.replace("/", "_").replace("-", "_")

    if args.local_update:
        print(f"Applying whitening_local_update ...")
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples,
                                    seed=args.seed, tokenizer=tokenizer,
                                    seqlen=args.model_seq_len)
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, dev)
        ckpt_path = os.path.join(args.save_path, f"{model_prefix}_v2_then_update_{keep}.pt")
    else:
        ckpt_path = os.path.join(args.save_path, f"{model_prefix}_v2_{keep}.pt")

    print(f"Saving checkpoint: {ckpt_path}")
    torch.save({"model": model, "tokenizer": tokenizer}, ckpt_path)
    print("Done.")


if __name__ == "__main__":
    main()
