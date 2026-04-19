#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/deac/csc/yangGrp/cuij/LLM/llm-pruner")
RESULTS_DIR = ROOT / "results"
PPL_DIR = RESULTS_DIR / "ppl_l31_8b" / "full"
SUMMARY_MD = Path("/deac/csc/yangGrp/cuij/LLM/results/compression_eval_summary.md")
SECTION_HEADER = "## Llama-3.1-8B / LLM-Pruner"

RETAIN_TO_PRUNE = {
    "1.0": None,
    "0.8": "0.2",
    "0.7": "0.3",
    "0.6": "0.4",
    "0.5": "0.5",
    "0.4": "0.6",
}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def mcq_metrics(payload: dict | None) -> tuple[str, str]:
    if not payload:
        return "", ""
    results = payload.get("results", {})
    if not results:
        return "", ""
    acc_vals = []
    report_vals = []
    for metrics in results.values():
        acc = metrics.get("acc")
        if acc is not None:
            acc_vals.append(float(acc))
        report_vals.append(float(metrics.get("acc_norm", metrics.get("acc"))))
    acc_mean = sum(acc_vals) / len(acc_vals) if acc_vals else None
    report_mean = sum(report_vals) / len(report_vals) if report_vals else None
    return (
        f"{acc_mean:.6f}" if acc_mean is not None else "",
        f"{report_mean:.6f}" if report_mean is not None else "",
    )


def ppl_metrics(payload: dict | None) -> tuple[str, str, str]:
    if not payload:
        return "", "", ""
    by_metric = payload.get("by_metric", {})
    wikitext2 = by_metric.get("wikitext2")
    c4 = by_metric.get("c4_stream")
    mean = payload.get("mean_ppl")
    return (
        f"{float(wikitext2):.6f}" if wikitext2 is not None else "",
        f"{float(c4):.6f}" if c4 is not None else "",
        f"{float(mean):.6f}" if mean is not None else "",
    )


def main() -> None:
    lines = [
        SECTION_HEADER,
        "",
        "| model | retain_ratio | wikitext2_ppl | c4_ppl | mean_ppl | mcq_acc_mean | mcq_report_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for retain_ratio, prune_ratio in RETAIN_TO_PRUNE.items():
        if prune_ratio is None:
            mcq_path = RESULTS_DIR / "llama31_8b_baseline_7task.json"
            ppl_path = PPL_DIR / "llama31_8b_baseline_ppl.json"
        else:
            mcq_path = RESULTS_DIR / f"llama31_8b_r{prune_ratio}_pruned_7task.json"
            ppl_path = PPL_DIR / f"llama31_8b_retain_{retain_ratio}_ppl.json"

        mcq_acc_mean, mcq_report_mean = mcq_metrics(load_json(mcq_path))
        wikitext2_ppl, c4_ppl, mean_ppl = ppl_metrics(load_json(ppl_path))
        lines.append(
            f"| llama31_8b | {retain_ratio} | {wikitext2_ppl} | {c4_ppl} | {mean_ppl} | {mcq_acc_mean} | {mcq_report_mean} |"
        )

    section = "\n".join(lines).rstrip() + "\n"
    if SUMMARY_MD.exists():
        text = SUMMARY_MD.read_text()
    else:
        text = ""

    if SECTION_HEADER in text:
        start = text.index(SECTION_HEADER)
        next_header = text.find("\n## ", start + len(SECTION_HEADER))
        if next_header == -1:
            text = text[:start].rstrip() + "\n\n" + section
        else:
            text = text[:start].rstrip() + "\n\n" + section + "\n" + text[next_header + 1 :].lstrip()
    else:
        text = text.rstrip() + "\n\n" + section if text.strip() else section

    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(text.rstrip() + "\n")
    print(SUMMARY_MD)


if __name__ == "__main__":
    main()
