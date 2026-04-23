from __future__ import annotations

import json
from pathlib import Path

from audit.run_audit import build_plan, run_feasibility


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_priority2_plan_builds_ifeval_command() -> None:
    plan = build_plan(PROJECT_ROOT / "audit" / "configs" / "2_ifeval.yaml")

    assert plan["audit_id"] == "2"
    assert plan["command_count"] == 1
    command = plan["commands"][0]
    assert command["stage"] == "eval"
    assert command["command"][:3] == ["python", "scripts/run_eval.py", "llama31-8b-instruct"]
    assert "instruct/ifeval" in command["command"]
    assert "--model-backend" in command["command"]


def test_priority4_plan_includes_compression_and_eval_commands() -> None:
    plan = build_plan(PROJECT_ROOT / "audit" / "configs" / "4_calibration_audit.yaml")

    assert plan["audit_id"] == "4"
    stages = [command["stage"] for command in plan["commands"]]
    assert "compress" in stages
    assert "eval" in stages
    assert any("calibration_profile=lm_only" in command["command"] for command in plan["commands"])
    assert any("alpaca_like" in command["shell"] for command in plan["commands"])


def test_priority3_plan_tracks_calibration_subsets() -> None:
    plan = build_plan(PROJECT_ROOT / "audit" / "configs" / "3_stability.yaml")

    assert plan["audit_id"] == "3"
    assert any("calibration_subset=subset1" in command["command"] for command in plan["commands"])
    assert any("calibration_offset=2048" in command["command"] for command in plan["commands"])


def test_priority1_feasibility_dry_run_writes_planned_result(tmp_path: Path) -> None:
    config = tmp_path / "a6.yaml"
    config.write_text(
        """
audit_id: "1"
kind: feasibility
run_label: test
feasibility:
  output_dir: ignored
  defaults:
    suite: memory/active
    device: cuda:0
    dtype: float16
    batch_size: 1
    prompt_length: 16
    generation_length: 4
  targets:
    - name: demo
      checkpoint: demo-checkpoint
      method: dobi_svd
      ratio: 0.6
""",
        encoding="utf-8",
    )

    summary = run_feasibility(config, output_dir=tmp_path / "out", dry_run=True)

    assert summary["result_count"] == 1
    assert summary["results"][0]["status"] == "planned"
    result_path = tmp_path / "out" / "results" / "feasibility__demo.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["checkpoint"] == "demo-checkpoint"
    assert "--prompt-length" in payload["command"]
