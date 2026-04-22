#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map requested keep ratios to BlockPruner del-block-num prefixes.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ppl-search-file", required=True)
    parser.add_argument("--targets", default="0.8,0.7,0.6,0.5,0.4")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def llama_profile(model_path: str) -> dict[str, int | bool]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if getattr(config, "model_type", None) != "llama":
        raise ValueError(f"Unsupported model_type={getattr(config, 'model_type', None)}; only llama is supported")

    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = hidden // heads
    kv_hidden = kv_heads * head_dim
    vocab = int(config.vocab_size)
    tie_embeddings = bool(getattr(config, "tie_word_embeddings", False))

    embed_params = vocab * hidden
    lm_head_params = 0 if tie_embeddings else vocab * hidden
    final_norm_params = hidden
    mha_block_params = (hidden * hidden) + (kv_hidden * hidden) + (kv_hidden * hidden) + (hidden * hidden) + hidden
    mlp_block_params = (intermediate * hidden) + (intermediate * hidden) + (hidden * intermediate) + hidden
    total_params = embed_params + lm_head_params + final_norm_params + layers * (mha_block_params + mlp_block_params)

    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "kv_hidden_size": kv_hidden,
        "vocab_size": vocab,
        "tie_word_embeddings": tie_embeddings,
        "embed_params": embed_params,
        "lm_head_params": lm_head_params,
        "final_norm_params": final_norm_params,
        "mha_block_params": mha_block_params,
        "mlp_block_params": mlp_block_params,
        "total_params": total_params,
    }


def load_prefixes(path: Path) -> dict[int, list[list[str | int]]]:
    payload = json.loads(path.read_text())
    prefixes: dict[int, list[list[str | int]]] = {}
    for key, value in payload.items():
        prefixes[int(key)] = value
    return prefixes


def prefix_stats(prefixes: dict[int, list[list[str | int]]], profile: dict[str, int | bool]) -> list[dict[str, object]]:
    total_params = int(profile["total_params"])
    mha_block_params = int(profile["mha_block_params"])
    mlp_block_params = int(profile["mlp_block_params"])

    stats: list[dict[str, object]] = []
    for del_block_num in sorted(prefixes):
        sequence = prefixes[del_block_num]
        seen: set[tuple[str, int]] = set()
        removed_params = 0
        mha_count = 0
        mlp_count = 0
        for block_type, block_id in sequence:
            block = (str(block_type), int(block_id))
            if block in seen:
                continue
            seen.add(block)
            if block[0] == "mha":
                removed_params += mha_block_params
                mha_count += 1
            elif block[0] == "mlp":
                removed_params += mlp_block_params
                mlp_count += 1
            else:
                raise ValueError(f"Unsupported block type: {block[0]}")
        keep_params = total_params - removed_params
        keep_ratio = keep_params / float(total_params)
        prune_ratio = removed_params / float(total_params)
        stats.append(
            {
                "del_block_num": del_block_num,
                "removed_params": removed_params,
                "kept_params": keep_params,
                "keep_ratio": keep_ratio,
                "prune_ratio": prune_ratio,
                "mha_blocks_removed": mha_count,
                "mlp_blocks_removed": mlp_count,
                "selected_blocks": sequence,
            }
        )
    return stats


def choose_targets(prefix_stats_payload: list[dict[str, object]], targets: list[float]) -> list[dict[str, object]]:
    selections: list[dict[str, object]] = []
    for target in targets:
        chosen = min(
            prefix_stats_payload,
            key=lambda item: (abs(float(item["keep_ratio"]) - target), int(item["del_block_num"])),
        )
        selections.append(
            {
                "target_keep_ratio": target,
                "selected_del_block_num": int(chosen["del_block_num"]),
                "achieved_keep_ratio": float(chosen["keep_ratio"]),
                "achieved_prune_ratio": float(chosen["prune_ratio"]),
                "absolute_error": abs(float(chosen["keep_ratio"]) - target),
                "mha_blocks_removed": int(chosen["mha_blocks_removed"]),
                "mlp_blocks_removed": int(chosen["mlp_blocks_removed"]),
                "selected_blocks": chosen["selected_blocks"],
            }
        )
    return selections


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    targets = [float(item.strip()) for item in args.targets.split(",") if item.strip()]
    profile = llama_profile(args.model_path)
    prefixes = load_prefixes(Path(args.ppl_search_file))
    stats = prefix_stats(prefixes, profile)
    selections = choose_targets(stats, targets)

    payload = {
        "model_path": args.model_path,
        "ppl_search_file": args.ppl_search_file,
        "targets": targets,
        "profile": profile,
        "prefix_stats": stats,
        "target_mapping": selections,
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n")

    print("target_keep_ratio\tselected_del_block_num\tachieved_keep_ratio\tachieved_prune_ratio\tabsolute_error")
    for item in selections:
        print(
            f"{item['target_keep_ratio']:.1f}\t{item['selected_del_block_num']}\t"
            f"{item['achieved_keep_ratio']:.6f}\t{item['achieved_prune_ratio']:.6f}\t{item['absolute_error']:.6f}"
        )


if __name__ == "__main__":
    main()
