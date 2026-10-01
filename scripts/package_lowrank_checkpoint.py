#!/usr/bin/env python3
"""
Generic LowRankArena checkpoint packager.

Typical use:
  python scripts/package_lowrank_checkpoint.py \
    --src outputs/aa_svd/llama31_8b_0.8/wikitext2_0.8 \
    --family llama31_8b \
    --method-dir aa_svd \
    --variant-dir objective2_refined_wikitext2_0.8 \
    --backbone llama-3.1-8b \
    --canonical-hf-path checkpoints/low_rank/llama31_8b/aa_svd/objective2_refined_wikitext2_0.8 \
    --native-val-ppl 9.684317588806152 \
    --wt2-ppl 14.816512311641297 \
    --c4-ppl 50.29615717678577 \
    --compress-log outputs/aa_svd/llama31_8b_0.8/aa_svd_compress.log \
    --expected-projections 224
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_TARGET_NAMES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


def expand_path(s):
    if s is None:
        return None
    return Path(os.path.expandvars(os.path.expanduser(s))).resolve()


def write_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def stage_checkpoint(src: Path, dst: Path, hardlink_weights: bool = True) -> None:
    """
    Copy a checkpoint into staging.
    By default, *.safetensors files are hard-linked when possible, avoiding
    duplication of multi-GB weight shards on the same filesystem.
    """
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name

        if item.is_dir():
            shutil.copytree(item, target)
            continue

        if hardlink_weights and item.suffix == ".safetensors":
            try:
                target.hardlink_to(item)
                continue
            except OSError:
                # Cross-filesystem or unsupported hardlink: fall back to copy.
                pass

        shutil.copy2(item, target)


def audit_structure(folder: Path, target_names, a_marker, b_marker):
    index_path = folder / "model.safetensors.index.json"
    idx = load_json(index_path)
    keys = list(idx["weight_map"].keys())

    a = [k for k in keys if a_marker in k]
    b = [k for k in keys if b_marker in k]
    dense = [
        k for k in keys
        if any(name in k for name in target_names)
        and a_marker not in k
        and b_marker not in k
    ]

    return {
        "a": len(a),
        "b": len(b),
        "dense": len(dense),
        "total": len(keys),
    }


def dtype_counts(folder: Path):
    counts = Counter()
    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"No *.safetensors files found in {folder}")

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                counts[str(sf.get_slice(key).get_dtype())] += 1

    return dict(sorted(counts.items()))


def validate_lowrank_config(folder: Path, strict: bool):
    cfg_path = folder / "config.json"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing config.json: {cfg_path}")

    cfg = load_json(cfg_path)

    if strict:
        if cfg.get("model_type") != "lowrank_llama":
            raise RuntimeError(
                f"Expected model_type=lowrank_llama, got {cfg.get('model_type')}"
            )

        arch = cfg.get("architectures")
        if arch != ["LowRankLlamaForCausalLM"]:
            raise RuntimeError(f"Unexpected architectures={arch}")

        auto_map = cfg.get("auto_map") or {}
        expected = {
            "AutoConfig": "configuration_lowrank_llama.LowRankLlamaConfig",
            "AutoModel": "modeling_lowrank_llama.LowRankLlamaModel",
            "AutoModelForCausalLM": "modeling_lowrank_llama.LowRankLlamaForCausalLM",
        }
        for key, value in expected.items():
            if auto_map.get(key) != value:
                raise RuntimeError(
                    f"auto_map[{key!r}]={auto_map.get(key)!r}, expected {value!r}"
                )

    return cfg


def copy_if_exists(src, dst):
    if src is not None and src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def build_parser():
    p = argparse.ArgumentParser(
        description="Package a LowRankArena factorized checkpoint for HF staging."
    )

    # Required source / destination identity
    p.add_argument("--src", required=True, help="Source checkpoint directory")
    p.add_argument("--family", required=True, help="e.g. llama_7b or llama31_8b")
    p.add_argument("--method-dir", required=True, help="e.g. aa_svd, gfw_svd")
    p.add_argument(
        "--variant-dir",
        required=True,
        help="e.g. objective2_refined_wikitext2_0.6",
    )
    p.add_argument("--backbone", required=True, help="Human-readable backbone name")
    p.add_argument(
        "--canonical-hf-path",
        required=True,
        help="Canonical path recorded inside metadata/README",
    )

    # Optional output root
    p.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "staging" / "upload"),
        help="Default: <repo>/staging/upload",
    )

    # Metrics
    p.add_argument("--native-val-ppl", type=float)
    p.add_argument("--wt2-ppl", type=float)
    p.add_argument("--c4-ppl", type=float)

    # Calibration
    p.add_argument("--calibration-corpus", default="wikitext2")
    p.add_argument("--calibration-samples", type=int, default=256)
    p.add_argument("--validation-samples", type=int, default=64)
    p.add_argument("--sequence-length", type=int, default=2048)
    p.add_argument("--native-seed", type=int, default=42)

    # Provenance
    p.add_argument("--compress-log")
    p.add_argument("--export-log")
    p.add_argument("--validation-log")

    # Structure
    p.add_argument("--expected-projections", type=int, default=224)
    p.add_argument("--a-marker", default=".ALinear.weight")
    p.add_argument("--b-marker", default=".BLinear.weight")
    p.add_argument(
        "--target-names",
        nargs="*",
        default=list(DEFAULT_TARGET_NAMES),
        help="Dense target projection suffixes to detect",
    )

    # Policy / labels
    p.add_argument("--expected-dtype", default="BF16")
    p.add_argument("--no-hardlink", action="store_true")
    p.add_argument("--no-strict-lowrank-config", action="store_true")
    p.add_argument("--dense-merge", action="store_true")
    p.add_argument("--quantization", action="store_true")
    p.add_argument("--remapping", action="store_true")
    p.add_argument("--external-recovery", action="store_true")
    p.add_argument(
        "--notes",
        default=None,
        help="Optional free-text note for manifest",
    )

    return p


def main():
    args = build_parser().parse_args()

    src = expand_path(args.src)
    out_root = expand_path(args.out_root)
    dst = out_root / args.family / args.method_dir / args.variant_dir

    compress_log = expand_path(args.compress_log)
    export_log = expand_path(args.export_log)
    validation_log = expand_path(args.validation_log)

    print("=" * 80)
    print("LowRankArena generic packager")
    print("=" * 80)
    print("SRC:", src)
    print("DST:", dst)

    if not src.exists():
        raise RuntimeError(f"Source checkpoint does not exist: {src}")

    required = [
        "config.json",
        "model.safetensors.index.json",
        "lowrankarena_method.json",
    ]
    for name in required:
        if not (src / name).exists():
            raise RuntimeError(f"Missing required file: {src / name}")

    stage_checkpoint(
        src,
        dst,
        hardlink_weights=not args.no_hardlink,
    )

    validate_lowrank_config(
        dst,
        strict=not args.no_strict_lowrank_config,
    )

    source_meta = load_json(src / "lowrankarena_method.json")

    requested = source_meta.get("requested_keep_ratio")
    achieved = source_meta.get("achieved_target_keep_ratio")

    if requested is None or achieved is None:
        raise RuntimeError(
            "lowrankarena_method.json must contain requested_keep_ratio "
            "and achieved_target_keep_ratio"
        )

    audit = audit_structure(
        dst,
        args.target_names,
        args.a_marker,
        args.b_marker,
    )

    print(
        "Structure:",
        f"A={audit['a']}",
        f"B={audit['b']}",
        f"dense={audit['dense']}",
        f"total={audit['total']}",
    )

    if audit["a"] != args.expected_projections:
        raise RuntimeError(
            f"Expected {args.expected_projections} A factors, got {audit['a']}"
        )

    if audit["b"] != args.expected_projections:
        raise RuntimeError(
            f"Expected {args.expected_projections} B factors, got {audit['b']}"
        )

    if not args.dense_merge and audit["dense"] != 0:
        raise RuntimeError(
            f"Expected 0 dense target projections, got {audit['dense']}"
        )

    dtypes = dtype_counts(dst)
    print("Dtype counts:", dtypes)

    expected_dtype = args.expected_dtype.upper()
    bad_dtypes = {
        dtype: count
        for dtype, count in dtypes.items()
        if dtype.upper() != expected_dtype
    }
    if bad_dtypes:
        raise RuntimeError(
            f"Expected all safetensors to be {expected_dtype}, got {bad_dtypes}"
        )

    # Preserve exporter metadata exactly.
    write_json(dst / "source_lowrankarena_method.json", source_meta)

    # Upload-level metadata.
    upload_meta = dict(source_meta)
    upload_meta.update(
        {
            "dense_merge_performed": bool(args.dense_merge),
            "hf_path": args.canonical_hf_path,
            "precision": expected_dtype.lower(),
            "storage_dtype": expected_dtype.lower(),
        }
    )
    write_json(dst / "lowrankarena_method.json", upload_meta)

    notes = args.notes
    if notes is None:
        notes = (
            "The checkpoint is structurally complete and remains factorized. "
            "Standardized LowRankArena evaluation metrics, when provided, "
            "are recorded separately from native-method validation metrics."
        )

    preliminary_results = {}
    if args.native_val_ppl is not None:
        preliminary_results["native_compression_output_ppl"] = args.native_val_ppl

    manifest = {
        "achieved_target_keep_ratio": achieved,
        "backbone": args.backbone,
        "calibration": {
            "corpus": args.calibration_corpus,
            "native_seed": args.native_seed,
            "sample_count": args.calibration_samples,
            "sequence_length": args.sequence_length,
        },
        "conversion": {
            "dense_merge": bool(args.dense_merge),
            "factor_multiplication": bool(args.dense_merge),
            "floating_tensors": (
                f"source checkpoint stored uniformly as {expected_dtype}"
            ),
            "quantization": bool(args.quantization),
        },
        "method": source_meta.get("method", args.method_dir),
        "notes": notes,
        "preliminary_results": preliminary_results,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_keep_ratio": requested,
        "schema_version": 1,
        "source_method_metadata": source_meta,
        "source_storage_dtype_counts": dtypes,
        "structure_audit": {
            "dense_merge_performed": bool(args.dense_merge),
            "dense_target_projections": audit["dense"],
            "factor_key_convention": (
                f"<projection>{args.a_marker} + <projection>{args.b_marker}"
            ),
            "factorized_target_projections": audit["a"],
            "target_projection_count": args.expected_projections,
        },
        "uploaded_storage_dtype_counts": dtypes,
        "validation_status": (
            "factorized structure and storage dtype audited"
        ),
    }

    standardized = {}
    if args.wt2_ppl is not None:
        standardized["wikitext2_ppl"] = args.wt2_ppl
    if args.c4_ppl is not None:
        standardized["c4_ppl"] = args.c4_ppl
    if standardized:
        manifest["standardized_lowrankarena_results"] = standardized

    write_json(dst / "lowrankarena_upload_manifest.json", manifest)

    # Provenance
    prov = dst / "provenance"
    prov.mkdir(exist_ok=True)

    copied_worker = copy_if_exists(compress_log, prov / "worker.log")
    copied_export = copy_if_exists(export_log, prov / "export.log")
    copied_validation = copy_if_exists(
        validation_log,
        prov / "validation_and_wiki_ppl.log",
    )

    if not copied_export:
        export_text = f"""\
LowRankArena staging/export summary
===================================

Source checkpoint: {src}
Staged checkpoint: {dst}
Prepared at UTC: {datetime.now(timezone.utc).isoformat()}

Method: {source_meta.get('method', args.method_dir)}
Backbone: {args.backbone}
Requested keep ratio: {requested}
Achieved keep ratio: {achieved}

Dense merge: {args.dense_merge}
Quantization: {args.quantization}
Remapping: {args.remapping}
External recovery: {args.external_recovery}

This file was generated by package_lowrank_checkpoint.py.
"""
        (prov / "export.log").write_text(export_text, encoding="utf-8")

    if not copied_validation:
        validation_lines = [
            "LowRankArena staging validation summary",
            "========================================",
            "",
            f"Model: {args.backbone}",
            f"Method: {source_meta.get('method', args.method_dir)}",
            f"Requested target keep ratio: {requested}",
            f"Achieved target keep ratio: {achieved}",
            "",
            "Structure audit",
            "---------------",
            f"A factors: {audit['a']}",
            f"B factors: {audit['b']}",
            f"Dense target projections: {audit['dense']}",
            f"Indexed tensors: {audit['total']}",
            f"Storage dtype counts: {json.dumps(dtypes)}",
            "",
            "Calibration",
            "-----------",
            f"Corpus: {args.calibration_corpus}",
            f"Calibration samples: {args.calibration_samples}",
            f"Validation samples: {args.validation_samples}",
            f"Sequence length: {args.sequence_length}",
            f"Native seed: {args.native_seed}",
        ]

        if args.native_val_ppl is not None:
            validation_lines += [
                "",
                "Native-method validation",
                "------------------------",
                f"Native validation PPL: {args.native_val_ppl}",
            ]

        if args.wt2_ppl is not None or args.c4_ppl is not None:
            validation_lines += [
                "",
                "Standardized LowRankArena evaluation",
                "------------------------------------",
            ]
            if args.wt2_ppl is not None:
                validation_lines.append(f"WikiText-2 PPL: {args.wt2_ppl}")
            if args.c4_ppl is not None:
                validation_lines.append(f"C4 PPL: {args.c4_ppl}")

        validation_lines += [
            "",
            "Notes",
            "-----",
            "This file is a staging summary assembled from supplied metadata/results.",
            "It is not raw evaluator stdout unless --validation-log was provided.",
        ]

        (prov / "validation_and_wiki_ppl.log").write_text(
            "\n".join(validation_lines) + "\n",
            encoding="utf-8",
        )

    if not copied_worker:
        (prov / "worker.log").write_text(
            "No raw compression/worker log was supplied to the packager.\n",
            encoding="utf-8",
        )

    # README with minimal valid HF model-card YAML.
    readme_lines = [
        "---",
        "library_name: transformers",
        "---",
        "",
        f"# {source_meta.get('method', args.method_dir)}: "
        f"{args.backbone}, keep {requested}",
        "",
        "This is a **factorized low-rank checkpoint**, not a dense reconstruction.",
        f"The `{args.a_marker}` and `{args.b_marker}` factors remain separate.",
        f"Floating tensors are stored uniformly in {expected_dtype}.",
        "",
        f"- Requested target-projection keep ratio: `{requested}`",
        f"- Achieved target-projection keep ratio: `{achieved}`",
        f"- Calibration: `{args.calibration_corpus}`, "
        f"`{args.calibration_samples} x {args.sequence_length}` tokens, "
        f"native seed `{args.native_seed}`",
        f"- Factorized/dense target projections: "
        f"`{audit['a']} / {audit['dense']}`",
    ]

    if args.native_val_ppl is not None:
        readme_lines.append(
            f"- Native validation PPL: `{args.native_val_ppl}`"
        )

    if args.wt2_ppl is not None or args.c4_ppl is not None:
        readme_lines += [
            "",
            "## Standardized LowRankArena evaluation",
            "",
        ]
        if args.wt2_ppl is not None:
            readme_lines.append(f"- WikiText-2 PPL: `{args.wt2_ppl}`")
        if args.c4_ppl is not None:
            readme_lines.append(f"- C4 PPL: `{args.c4_ppl}`")

    readme_lines += [
        "",
        "## Loading",
        "",
        "```python",
        "import torch",
        "from transformers import AutoModelForCausalLM",
        "",
        "model = AutoModelForCausalLM.from_pretrained(",
        '    "Duke-CEI-SVD/LowRankArena",',
        f'    subfolder="{args.canonical_hf_path}",',
        "    trust_remote_code=True,",
        "    torch_dtype=torch.bfloat16,",
        '    device_map="auto",',
        ")",
        "```",
        "",
    ]

    (dst / "README.md").write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )

    print()
    print("PASS")
    print("  method          :", source_meta.get("method", args.method_dir))
    print("  requested keep  :", requested)
    print("  achieved keep   :", achieved)
    print("  A/B/dense       :", audit["a"], audit["b"], audit["dense"])
    print("  dtype counts    :", dtypes)
    if args.native_val_ppl is not None:
        print("  native val PPL  :", args.native_val_ppl)
    if args.wt2_ppl is not None:
        print("  WT2 PPL         :", args.wt2_ppl)
    if args.c4_ppl is not None:
        print("  C4 PPL          :", args.c4_ppl)
    print("  output          :", dst)


if __name__ == "__main__":
    main()
