# Calibration-Data Sensitivity

These audits isolate questions about calibration corpus identity, selected
source units, sample count, and variation in downstream measurements.

- [`calibration_source/`](./calibration_source/README.md) verifies the frozen,
  method-independent source-indexed inputs.
- [`rank_stability/`](./rank_stability/README.md) recomputes the observed
  four-method ordering across three WikiText draws and one C4 draw.

The audits consume published artifacts and do not regenerate checkpoints.
