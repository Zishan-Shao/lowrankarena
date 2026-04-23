# `audit/`

This directory holds secondary-audit experiment definitions and lightweight orchestration helpers. It is intentionally separate from `benchmark/`: the main benchmark lanes stay stable, while appendix audits can compose compression, evaluation, and feasibility probes without becoming primary claims.

## Included Audits

- [`configs/1_large_model_feasibility.yaml`](./configs/1_large_model_feasibility.yaml): Table A1 feasibility probes for selected 13B/70B methods. Disabled placeholder targets are included until exact large-model checkpoints are registered.
- [`configs/2_ifeval.yaml`](./configs/2_ifeval.yaml): Appendix IFEval instruction-following audit over selected instruct checkpoints.
- [`configs/3_stability.yaml`](./configs/3_stability.yaml): Table A2 calibration subset / seed stability plan.
- [`configs/4_calibration_audit.yaml`](./configs/4_calibration_audit.yaml): Figure 7 LM-only vs Alpaca-like calibration plan.

## Common Workflow

Build a reproducible command plan:

```bash
python audit/run_audit.py plan audit/configs/2_ifeval.yaml \
  --output results/audit/2_ifeval/plan.json \
  --script results/audit/2_ifeval/jobs.sh
```

Run the generated plan:

```bash
bash results/audit/2_ifeval/jobs.sh
```

For priority 1, use the specialized runner so OOM and other failures are captured as normalized audit JSON rather than disappearing into logs:

```bash
python audit/run_audit.py run-feasibility audit/configs/1_large_model_feasibility.yaml \
  --include-disabled \
  --only llama70b-dobi-r60
```

It writes per-target JSON files plus `summary.json` and `table.md` under `results/audit/1_large_model_feasibility/`.

## Priority 3 and 4 Notes

The priority 3/4 configs generate both compression commands and follow-up eval commands. The compression commands pass audit metadata through `--extra`, for example `calibration_profile`, `calibration_subset`, and `calibration_offset`.

After artifacts are materialized, register the exact checkpoint names expected by each config's `evaluation.checkpoint_template`, or edit the template to match the registered IDs. Then run the eval commands from the generated plan.

Summaries:

```bash
python audit/run_audit.py summarize-calibration audit/configs/4_calibration_audit.yaml \
  --output-dir results/audit/4_calibration

python audit/run_audit.py summarize-stability audit/configs/3_stability.yaml \
  --output-dir results/audit/3_stability
```

The summaries read normalized eval JSON from each config's `evaluation.output_dir`.
