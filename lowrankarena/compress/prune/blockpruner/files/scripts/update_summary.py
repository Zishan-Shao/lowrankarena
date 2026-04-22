#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/deac/csc/yangGrp/cuij/LLM/BlockPruner")
SUMMARY_MD = Path("/deac/csc/yangGrp/cuij/LLM/results/compression_eval_summary.md")
FORMAL_ROOT = ROOT / "results" / "formal"
TARGETS = ["0.8", "0.7", "0.6", "0.5", "0.4"]
TASK_NOTE = "Standardized MCQ here is the 7-task suite: `openbookqa`, `arc_easy`, `arc_challenge`, `piqa`, `winogrande`, `hellaswag`, `boolq`."
CALIB_NOTE = "Requested retain ratios are mapped to the nearest achievable prefix in the BlockPruner deletion order; see the calibration JSON in the corresponding formal result directory."

MODELS = [
    {
        "header": "## Llama-3.1-8B / BlockPruner",
        "model_id": "llama31_8b",
        "result_key": "llama31_8b",
    },
    {
        "header": "## Llama-1-7B / BlockPruner",
        "model_id": "llama1_7b",
        "result_key": "llama1_7b",
    },
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def ppl_metrics(payload: dict | None) -> tuple[str, str, str]:
    if not payload:
        return "-", "-", "-"
    by_metric = payload.get("by_metric", {})
    wikitext2 = by_metric.get("wikitext2")
    c4 = by_metric.get("c4_stream")
    mean = payload.get("mean_ppl")
    return (
        f"{float(wikitext2):.6f}" if wikitext2 is not None else "-",
        f"{float(c4):.6f}" if c4 is not None else "-",
        f"{float(mean):.6f}" if mean is not None else "-",
    )


def mcq_metrics(payload: dict | None) -> tuple[str, str]:
    if not payload:
        return "-", "-"
    acc_mean = payload.get("mcq_acc_mean")
    report_mean = payload.get("mcq_report_mean")
    return (
        f"{float(acc_mean):.6f}" if acc_mean is not None else "-",
        f"{float(report_mean):.6f}" if report_mean is not None else "-",
    )


def replace_section(text: str, header: str, section: str) -> str:
    if header in text:
        start = text.index(header)
        next_header = text.find("\n## ", start + len(header))
        if next_header == -1:
            return text[:start].rstrip() + "\n\n" + section
        return text[:start].rstrip() + "\n\n" + section + "\n" + text[next_header + 1 :].lstrip()
    return text.rstrip() + "\n\n" + section if text.strip() else section


def build_section(model: dict[str, str]) -> str:
    result_dir = FORMAL_ROOT / model["result_key"] / "standardized"
    calib_json = FORMAL_ROOT / model["result_key"] / "calibration" / f"{model['result_key']}_keep_ratio_calibration.json"
    lines = [
        model["header"],
        "",
        CALIB_NOTE,
        TASK_NOTE,
        f"Calibration JSON: `{calib_json}`",
        "",
        "| model | retain_ratio | wikitext2_ppl | c4_ppl | mean_ppl | mcq_acc_mean | mcq_report_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for retain in TARGETS:
        ppl_path = result_dir / f"{model['result_key']}_keep_{retain}_ppl.json"
        mcq_path = result_dir / f"{model['result_key']}_keep_{retain}_7task.json"
        wikitext2_ppl, c4_ppl, mean_ppl = ppl_metrics(load_json(ppl_path))
        mcq_acc_mean, mcq_report_mean = mcq_metrics(load_json(mcq_path))
        lines.append(
            f"| {model['model_id']} | {retain} | {wikitext2_ppl} | {c4_ppl} | {mean_ppl} | {mcq_acc_mean} | {mcq_report_mean} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    text = SUMMARY_MD.read_text() if SUMMARY_MD.exists() else ""
    for model in MODELS:
        text = replace_section(text, model["header"], build_section(model))
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(text.rstrip() + "\n")
    print(SUMMARY_MD)


if __name__ == "__main__":
    main()
