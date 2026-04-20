"""
SVD-LLM V2 heterogeneous compression runner.
Uses SVDLLM_v2_hetero.py for adaptive per-module rank allocation + eigendecomp profiling.
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from SVDLLM_v2_hetero import (
    profle_svdllm as hetero_profile,
    whitening_hetero,
    allocate_svdllm_v2_adaptive_keep_ratios,
)
from SVDLLM import whitening_local_update
from utils.data_utils import get_loaders, get_calib_train_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_seq_len", type=int, default=2048)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--svd_method", type=str, default="full", choices=["full", "randomized"])
    parser.add_argument("--no_adaptive", action="store_true",
                        help="Skip adaptive rank allocation, use uniform ratio.")
    # Profiling options (mutually exclusive)
    prof_group = parser.add_mutually_exclusive_group(required=True)
    prof_group.add_argument("--profiling_mat_path", type=str, default=None,
                            help="Path to existing profiling_mat (Cholesky or eigendecomp format).")
    prof_group.add_argument("--reprofile", action="store_true",
                            help="Re-profile from scratch using hetero eigendecomp profiler.")
    parser.add_argument("--calib_dataset", type=str, default="wikitext2")
    parser.add_argument("--calib_nsamples", type=int, default=256)
    parser.add_argument("--save_profiling_mat", type=str, default=None,
                        help="If set, save the new profiling_mat to this path (only with --reprofile).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model.split("/")[-1].lower()

    print(f"Loading model: {args.model}")
    hf_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16}
    if args.hf_token:
        hf_kwargs["token"] = args.hf_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **hf_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                              **({"token": args.hf_token} if args.hf_token else {}))
    model.eval()
    model.seqlen = args.model_seq_len

    if args.reprofile:
        print(f"Profiling from scratch (hetero eigendecomp, dataset={args.calib_dataset}, n={args.calib_nsamples}) ...")
        calib_loader = get_calib_train_data(args.calib_dataset, tokenizer,
                                            args.calib_nsamples, seqlen=args.model_seq_len,
                                            seed=args.seed)
        profiling_mat = hetero_profile(model_name, model, calib_loader, dev)
        if args.save_profiling_mat:
            os.makedirs(os.path.dirname(args.save_profiling_mat) or ".", exist_ok=True)
            torch.save(profiling_mat, args.save_profiling_mat)
            print(f"Profiling_mat saved: {args.save_profiling_mat}")
    else:
        print(f"Loading profiling_mat: {args.profiling_mat_path}")
        profiling_mat = torch.load(args.profiling_mat_path, weights_only=False)

    model = model.to(dev)

    module_keep_ratios = None
    if not args.no_adaptive:
        print(f"Adaptive rank allocation (target_reduction={args.ratio}) ...")
        module_keep_ratios, _, _ = allocate_svdllm_v2_adaptive_keep_ratios(
            model_name=model_name,
            model=model,
            profiling_mat=profiling_mat,
            target_reduction_ratio=args.ratio,
            dev=dev,
        )

    ratio_keep = 1.0 - args.ratio
    print(f"Compressing (ratio_keep={ratio_keep}) ...")
    whitening_hetero(
        model_name=model_name,
        model=model,
        profiling_mat=profiling_mat,
        ratio=ratio_keep,
        dev=dev,
        svd_method=args.svd_method,
        module_keep_ratios=module_keep_ratios,
    )

    os.makedirs(args.save_path, exist_ok=True)
    keep = 1 - args.ratio
    model_prefix = args.model.replace("/", "_").replace("-", "_")
    suffix = "v2hetero" if not args.no_adaptive else "v2hetero_uniform"
    ckpt_path = os.path.join(args.save_path, f"{model_prefix}_{suffix}_{keep}.pt")
    print(f"Saving: {ckpt_path}")
    torch.save({"model": model, "tokenizer": tokenizer}, ckpt_path)
    print("Done.")


if __name__ == "__main__":
    main()
