# coding:utf8
import os
import sys
import time
import argparse
from itertools import islice

import torch

from utils.model_utils import get_model_from_huggingface, get_model_from_local
from utils.data_utils import get_test_data


'''
Original model:
CUDA_VISIBLE_DEVICES=7 python svdllm_gen.py --model openlm-research/open_llama_7b --model_path original --prefill_len 2048 --batch_size 4 --num_batches 5

Local SVD checkpoint:
CUDA_VISIBLE_DEVICES=7 python svdllm_gen.py --model_path ./jeffwan_llama_7b_hf_whitening_only_0.4.pt --prefill_len 2048 --batch_size 4 --num_batches 5

'''

def to_mib(x_bytes: int) -> float:
    return x_bytes / (1024.0 ** 2)


@torch.no_grad()
def profile_prefill_decode(model, tokenizer, dataset: str, prefill_len: int, decode_len: int, batch_size: int,
                           device: str, num_batches: int = 5):
    model.eval()
    model.to(device)

    # Record parameter memory footprint on device
    weight_mem_mib = 0.0
    peak_alloc_mib = 0.0
    peak_reserved_mib = 0.0
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        weight_mem_mib = to_mib(torch.cuda.memory_allocated())
        torch.cuda.empty_cache()

    test_loader = get_test_data(dataset, tokenizer, seq_len=prefill_len, batch_size=batch_size)

    agg = {
        'prefill_time_s': 0.0,
        'decode_time_s': 0.0,
        'batches': 0,
        'tokens_generated': 0,
        'prefill_peak_alloc_mib': 0.0,
        'prefill_peak_reserved_mib': 0.0,
        'decode_peak_alloc_mib': 0.0,
        'decode_peak_reserved_mib': 0.0,
    }

    def get_peaks():
        return to_mib(torch.cuda.max_memory_allocated()), to_mib(torch.cuda.max_memory_reserved())

    for batch_idx, input_ids in enumerate(islice(test_loader, num_batches)):
        input_ids = input_ids.to(device)

        # Prefill phase (prompt forward pass, building KV cache)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(
            input_ids=input_ids,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            prefill_peak_alloc_mib, prefill_peak_reserved_mib = get_peaks()
        else:
            prefill_peak_alloc_mib = prefill_peak_reserved_mib = 0.0

        past_kv = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1)

        # Decode phase (step-by-step generation with KV cache)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t2 = time.time()

        total_steps = decode_len
        tokens_gen = 0
        for _ in range(total_steps):
            out = model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
            tokens_gen += input_ids.shape[0]

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t3 = time.time()

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            decode_peak_alloc_mib, decode_peak_reserved_mib = get_peaks()
        else:
            decode_peak_alloc_mib = decode_peak_reserved_mib = 0.0

        agg['batches'] += 1
        agg['prefill_time_s'] += (t1 - t0)
        agg['decode_time_s'] += (t3 - t2)
        agg['tokens_generated'] += tokens_gen
        agg['prefill_peak_alloc_mib'] = max(agg['prefill_peak_alloc_mib'], prefill_peak_alloc_mib)
        agg['prefill_peak_reserved_mib'] = max(agg['prefill_peak_reserved_mib'], prefill_peak_reserved_mib)
        agg['decode_peak_alloc_mib'] = max(agg['decode_peak_alloc_mib'], decode_peak_alloc_mib)
        agg['decode_peak_reserved_mib'] = max(agg['decode_peak_reserved_mib'], decode_peak_reserved_mib)

    # Summarize
    avg_prefill_time = agg['prefill_time_s'] / max(agg['batches'], 1)
    avg_decode_time = agg['decode_time_s'] / max(agg['batches'], 1)
    throughput_tok_s = agg['tokens_generated'] / max(agg['decode_time_s'], 1e-6)

    print("Memory/Throughput Summary")
    print("Weight Memory: {:.2f} MiB".format(weight_mem_mib))
    print("Prefill Peak (allocated): {:.2f} MiB".format(agg['prefill_peak_alloc_mib']))
    print("Prefill Peak (reserved):  {:.2f} MiB".format(agg['prefill_peak_reserved_mib']))
    print("Decode Peak (allocated):  {:.2f} MiB".format(agg['decode_peak_alloc_mib']))
    print("Decode Peak (reserved):   {:.2f} MiB".format(agg['decode_peak_reserved_mib']))
    print("Avg Prefill Time / batch: {:.3f} s".format(avg_prefill_time))
    print("Avg Decode Time / batch:  {:.3f} s ({} new tokens)".format(avg_decode_time, decode_len))
    print("Decode Throughput:        {:.2f} tok/s".format(throughput_tok_s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openlm-research/open_llama_7b', help='HF model id')
    parser.add_argument('--model_path', type=str, default='original', help='"original" or local checkpoint .pt path')
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Dataset for prompts [wikitext2, ptb, c4]')
    parser.add_argument('--prefill_len', type=int, default=2048, help='Prompt length (tokens)')
    parser.add_argument('--decode_len', type=int, default=128, help='Number of new tokens to decode')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for prompts')
    parser.add_argument('--num_batches', type=int, default=5, help='How many batches to profile')
    parser.add_argument('--device', type=str, default='cuda', help='Device e.g., cuda or cuda:0')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face access token (optional)')

    args = parser.parse_args()

    if args.model_path == 'original':
        model, tokenizer = get_model_from_huggingface(args.model, hf_token=args.hf_token)
    else:
        model, tokenizer = get_model_from_local(args.model_path)

    # Keep dtype as loaded; user can cast externally if desired
    profile_prefill_decode(
        model=model,
        tokenizer=tokenizer,
        dataset=args.dataset,
        prefill_len=args.prefill_len,
        decode_len=args.decode_len,
        batch_size=args.batch_size,
        device=args.device,
        num_batches=args.num_batches,
    )


if __name__ == '__main__':
    main()

