# LowRankArena Audits

This directory contains small public reproductions for claims that are easier
to inspect separately from the main benchmark. Large inputs, checkpoints, and
raw result collections remain in
[`Duke-CEI-SVD/LowRankArena`](https://huggingface.co/Duke-CEI-SVD/LowRankArena)
on Hugging Face. Each audit pins the exact HF revision that it consumes.

The stable taxonomy is:

```text
audits/
├── calibration_data_sensitivity/
├── implementation_readiness/
└── inference_sensitivity/
```

- [`calibration_data_sensitivity/`](./calibration_data_sensitivity/README.md)
  covers calibration source, selected samples, sample count, and the resulting
  metric or rank sensitivity.
- [`implementation_readiness/`](./implementation_readiness/README.md) covers
  source provenance, loaders, adapters, checkpoint structure, and runtime
  compatibility.
- [`inference_sensitivity/`](./inference_sensitivity/README.md) covers task,
  workload, hardware, backend, kernel, TTFT, ITL, and throughput sensitivity.

No model compression or checkpoint generation is required by the audits added
here. Internal author-response working directories are intentionally excluded
from Git; only the minimal public evidence and reproduction code belongs here.

See [`manifest.json`](./manifest.json) for machine-readable audit discovery.
