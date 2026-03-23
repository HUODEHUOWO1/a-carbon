# Measurement Contract (Frozen)

This project uses a frozen measurement stack for real profiling.

## Frozen config

- Contract: `measurement_stack_frozen_v1`
- Config source: `configs/measurement_contract.yaml`
- Default profile config: `configs/profiling_real_resnet50.yaml`
- Hardware scope: `RTX4060_Laptop`

## Canonical columns

- Canonical latency column: `latency_per_infer_ms`
- Canonical energy column: `energy_per_infer_Wh`

Legacy aliases (for compatibility only):

- `latency_ms` aliases `latency_per_infer_ms`
- `energy_Wh` aliases `energy_per_infer_Wh`

Downstream analysis should always read canonical columns.

## Required window/audit columns

- `latency_window_ms`
- `energy_window_Wh`
- `energy_window_Wh_powerint`
- `implied_power_W`
- `energy_backend_rel_err`
- `energy_backend_signed_err`

## Frozen measurement params

- `repeat_per_request = 400`
- `energy_backend = nvml_total_energy`
- `power_integration_enabled = true`
- `power_sample_interval_ms = 10`
- `power_min_samples = 20`

## Change policy

Do not change measurement semantics during formal profiling.
Any change to backend/repeat/window requires re-running gate audits.


## TensorRT spike status (proxy)

- Contract: `configs/measurement_contract_tensorrt_spike.yaml`
- Audit report: `runs/spike/vision_tensorrt_energy_backend_audit.json`
- Decision: use `nvml_total_energy` as canonical backend; keep power integration as diagnostic only for this spike stage.
- Scope: proxy-only (`paper_use=false`), not formal precomputed input.
