#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/deac/csc/yangGrp/cuij/LLM/llm-pruner")
RESULTS_DIR = ROOT / "results"
PPL_DIR = RESULTS_DIR / "ppl_l1_7b" / "full"
SUMMARY_MD = Path("/deac/csc/yangGrp/cuij/LLM/results/compression_eval_summary.md")

SLICEGPT_ROOT = Path("/deac/csc/yangGrp/cuij/LLM/TransformerCompression")
SLICEGPT_RESULTS_DIR = SLICEGPT_ROOT / "results" / "formal_l1_7b"
SLICEGPT_LOG_DIR = SLICEGPT_ROOT / "logs" / "local_l1_7b"
HAPE_ROOT = Path("/deac/csc/yangGrp/cuij/LLM/HAP-E")
HAPE_EVAL_DIR = HAPE_ROOT / "eval_results" / "llama1_7b"
HAPE_PLACEHOLDER_MARKER = HAPE_EVAL_DIR / ".placeholder_until_rerun_complete"

RETAIN_TO_PRUNE = {
    "0.8": "0.2",
    "0.7": "0.3",
    "0.6": "0.4",
    "0.5": "0.5",
    "0.4": "0.6",
}

LLM_PRUNER_SECTION_HEADER = "## Llama-1-7B / LLM-Pruner"
SLICEGPT_SECTION_HEADER = "## Llama-1-7B / SliceGPT"
HAPE_SECTION_HEADER = "## Llama-1-7B / HAP-E"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def llm_pruner_mcq_metrics(payload: dict | None) -> tuple[str, str]:
    if not payload:
        return "-", "-"
    results = payload.get("results", {})
    if not results:
        return "-", "-"
    acc_vals = []
    report_vals = []
    for metrics in results.values():
        acc = metrics.get("acc")
        if acc is not None:
            acc_vals.append(float(acc))
        report = metrics.get("acc_norm", metrics.get("acc"))
        if report is not None:
            report_vals.append(float(report))
    acc_mean = sum(acc_vals) / len(acc_vals) if acc_vals else None
    report_mean = sum(report_vals) / len(report_vals) if report_vals else None
    return (
        f"{acc_mean:.6f}" if acc_mean is not None else "-",
        f"{report_mean:.6f}" if report_mean is not None else "-",
    )


def llm_pruner_ppl_metrics(payload: dict | None) -> tuple[str, str, str]:
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


def slicegpt_mcq_metrics(payload: dict | None) -> tuple[str, str]:
    if not payload:
        return "-", "-"
    acc_vals = []
    report_vals = []
    for metrics in payload.values():
        acc = metrics.get("acc,none")
        if acc is not None:
            acc_vals.append(float(acc))
        report = metrics.get("acc_norm,none", metrics.get("acc,none"))
        if report is not None:
            report_vals.append(float(report))
    acc_mean = sum(acc_vals) / len(acc_vals) if acc_vals else None
    report_mean = sum(report_vals) / len(report_vals) if report_vals else None
    return (
        f"{acc_mean:.6f}" if acc_mean is not None else "-",
        f"{report_mean:.6f}" if report_mean is not None else "-",
    )


def slicegpt_wikitext2_ppl(retain: str) -> str:
    ratio_tag = retain.replace(".", "")
    candidates = sorted(SLICEGPT_LOG_DIR.glob(f"slicegpt_l1_k{ratio_tag}_g*.log"))
    if not candidates:
        return "-"
    text = candidates[-1].read_text(errors="ignore")
    match = re.search(r"After rotating and slicing ([0-9.]+)", text)
    return f"{float(match.group(1)):.6f}" if match else "-"


def slicegpt_c4_ppl(retain: str) -> str:
    payload = load_json(SLICEGPT_RESULTS_DIR / f"keep_{retain}" / "eval_ppl_c4" / "metrics.json")
    if not payload:
        return "-"
    ppl = payload.get("ppl")
    return f"{float(ppl):.6f}" if ppl is not None else "-"


def mean_ppl_text(wikitext2_ppl: str, c4_ppl: str) -> str:
    if "-" in {wikitext2_ppl, c4_ppl}:
        return "-"
    return f"{(float(wikitext2_ppl) + float(c4_ppl)) / 2.0:.6f}"


def build_llm_pruner_section() -> str:
    lines = [
        LLM_PRUNER_SECTION_HEADER,
        "",
        "| model | retain_ratio | wikitext2_ppl | c4_ppl | mean_ppl | mcq_acc_mean | mcq_report_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline_mcq = RESULTS_DIR / "llama1_7b_baseline_7task.json"
    baseline_ppl = PPL_DIR / "llama1_7b_baseline_ppl.json"
    mcq_acc_mean, mcq_report_mean = llm_pruner_mcq_metrics(load_json(baseline_mcq))
    wikitext2_ppl, c4_ppl, mean_ppl = llm_pruner_ppl_metrics(load_json(baseline_ppl))
    lines.append(
        f"| llama1_7b | 1.0 | {wikitext2_ppl} | {c4_ppl} | {mean_ppl} | {mcq_acc_mean} | {mcq_report_mean} |"
    )
    for retain, prune in RETAIN_TO_PRUNE.items():
        mcq_path = RESULTS_DIR / f"llama1_7b_r{prune}_pruned_7task.json"
        ppl_path = PPL_DIR / f"llama1_7b_retain_{retain}_ppl.json"
        mcq_acc_mean, mcq_report_mean = llm_pruner_mcq_metrics(load_json(mcq_path))
        wikitext2_ppl, c4_ppl, mean_ppl = llm_pruner_ppl_metrics(load_json(ppl_path))
        lines.append(
            f"| llama1_7b | {retain} | {wikitext2_ppl} | {c4_ppl} | {mean_ppl} | {mcq_acc_mean} | {mcq_report_mean} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_slicegpt_section() -> str:
    lines = [
        SLICEGPT_SECTION_HEADER,
        "",
        "Repo-default SliceGPT MCQ here is the 5-task suite: `piqa`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`.",
        "This section is refreshed from local SliceGPT result files as they land.",
        "",
        "| model | retain_ratio | wikitext2_ppl | c4_ppl | mean_ppl | mcq_acc_mean | mcq_report_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for retain in RETAIN_TO_PRUNE:
        mcq_path = SLICEGPT_RESULTS_DIR / f"keep_{retain}" / "eval_mcq" / "full_results_0_shot.json"
        wikitext2_ppl = slicegpt_wikitext2_ppl(retain)
        c4_ppl = slicegpt_c4_ppl(retain)
        mean_ppl = mean_ppl_text(wikitext2_ppl, c4_ppl)
        mcq_acc_mean, mcq_report_mean = slicegpt_mcq_metrics(load_json(mcq_path))
        lines.append(
            f"| llama1_7b | {retain} | {wikitext2_ppl} | {c4_ppl} | {mean_ppl} | {mcq_acc_mean} | {mcq_report_mean} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_hape_section() -> str:
    lines = [
        HAPE_SECTION_HEADER,
        "",
        "HAP-E entries are evaluated from the dense `pruned_model.bin` state_dict against the base `huggyllama__llama-7b` model.",
    ]
    if HAPE_PLACEHOLDER_MARKER.exists():
        lines.extend(
            [
                "Current HAP-E rows are intentionally hidden with placeholders while the fixed rerun is in progress.",
                "Do not use the older canonical HAP-E numbers below; they came from pre-fix runs.",
                "",
                "| model | retain_ratio | wikitext2_ppl | c4_ppl | mean_ppl | mcq_acc_mean | mcq_report_mean |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for retain in RETAIN_TO_PRUNE:
            lines.append(f"| llama1_7b | {retain} | - | - | - | - | - |")
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(
        [
            "",
            "| model | retain_ratio | wikitext2_ppl | c4_ppl | mean_ppl | mcq_acc_mean | mcq_report_mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for retain in RETAIN_TO_PRUNE:
        mcq_path = HAPE_EVAL_DIR / f"llama1_7b_keep_{retain}_7task.json"
        ppl_path = HAPE_EVAL_DIR / f"llama1_7b_keep_{retain}_ppl.json"
        mcq_acc_mean, mcq_report_mean = llm_pruner_mcq_metrics(load_json(mcq_path))
        wikitext2_ppl, c4_ppl, mean_ppl = llm_pruner_ppl_metrics(load_json(ppl_path))
        lines.append(
            f"| llama1_7b | {retain} | {wikitext2_ppl} | {c4_ppl} | {mean_ppl} | {mcq_acc_mean} | {mcq_report_mean} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def replace_section(text: str, header: str, section: str) -> str:
    if header in text:
        start = text.index(header)
        next_header = text.find("\n## ", start + len(header))
        if next_header == -1:
            return text[:start].rstrip() + "\n\n" + section
        return text[:start].rstrip() + "\n\n" + section + "\n" + text[next_header + 1 :].lstrip()
    return text.rstrip() + "\n\n" + section if text.strip() else section


def main() -> None:
    if SUMMARY_MD.exists():
        text = SUMMARY_MD.read_text()
    else:
        text = ""

    text = replace_section(text, LLM_PRUNER_SECTION_HEADER, build_llm_pruner_section())
    text = replace_section(text, SLICEGPT_SECTION_HEADER, build_slicegpt_section())
    text = replace_section(text, HAPE_SECTION_HEADER, build_hape_section())

    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(text.rstrip() + "\n")
    print(SUMMARY_MD)


if __name__ == "__main__":
    main()
