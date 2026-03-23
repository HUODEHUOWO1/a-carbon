# CAI ICWS 2026 Frozen Execution Workflow

## 0) Install

```powershell
cd E:\a-carbon
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If editable install fails, run with source path:

```powershell
$env:PYTHONPATH="E:\a-carbon\src"
python -m cai_lab.cli --help
```


## Frozen Notes (Current)

- Formal matrix is precision-only: `capacity_k` is disabled in formal experiments.
- Capacity gate outputs are archived negative results and must not be used for formal precomputed export.
- Measurement stack is frozen by `configs/measurement_contract.yaml`.

## 1) Freeze Matrix

```powershell
python -m cai_lab.cli freeze-matrix --matrix configs/experiment_matrix.yaml --output runs/frozen_matrix.yaml
```

## 2) Real Profiling Pipeline (No Electricity Maps Needed)

`real profiling` is generated from live GPU inference, not downloaded data.

### 2.1 Prepare Config

Use one of these templates and fill real paths:
- `configs/profiling_real_resnet50.example.yaml`
- `configs/profiling_real_bert.example.yaml`

Copy to active config file, for example:

```powershell
copy configs\profiling_real_resnet50.example.yaml configs\profiling_real_resnet50.yaml
```

### 2.2 Run Minimal Real Chain (recommended first)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_real_minimal.ps1 -Config configs/profiling_real_resnet50.yaml
```

Equivalent direct commands:

```powershell
python -m cai_lab.cli profile-real-run --config configs/profiling_real_resnet50.yaml --mode-ids fp16_k4 --allow-partial-modes
python -m cai_lab.cli profile-real-validate --config configs/profiling_real_resnet50.yaml
python -m cai_lab.cli profile-real-export --config configs/profiling_real_resnet50.yaml --output-root runs/input_profiles
```

### 2.3 Output Layout

Live outputs are written under:

```text
artifacts/profiling_real/<workload_id>/<gpu_name>/bs1/
```

Per mode:
- `request_metrics.csv`
- `summary.json`
- `env.json`
- `sample_ids.csv`

Workload-level curated files:
- `mode_summary.csv`
- `admitted_modes.csv`
- `mode_status.csv`
- `manifest.yaml`

Switch penalty:
- `switch_penalty/<modeA>__<modeB>.csv`
- `switch_penalty/summary.csv`
- `switch_penalty/switch_penalties.yaml`

### 2.4 Minimal Acceptance Files

At minimum, verify these files exist and pass schema checks:
1. `<...>/fp16_k4/request_metrics.csv`
2. `<...>/int8_k4/request_metrics.csv`
3. `<...>/mode_summary.csv`
4. `<...>/admitted_modes.csv`
5. `<...>/switch_penalty/summary.csv`
6. `<...>/fp16_k4/env.json`

### 2.5 Notes

- Built-in live runtime supports:
  - `backend=pytorch` (`fp32/fp16/bf16`)
  - `backend=tensorrt` for `vision_classification` (`fp16/int8/fp32`)
- Keep `backend=tensorrt` scope limited to the implemented workload/runtime path; do not fake unsupported modes.
- Energy backend `nvml_total_energy` requires `pynvml` (`pip install nvidia-ml-py3`).

## 3) Existing E0 Pipeline (from precomputed root)

After `profile-real-export`, run the existing E0 chain using real data:

```powershell
python -m cai_lab.cli e0-profile --matrix runs/frozen_matrix.yaml --output runs/e0/profiling --precomputed-root runs/input_profiles --strict-precomputed
python -m cai_lab.cli e0-prune --matrix runs/frozen_matrix.yaml --profile-summary runs/e0/profiling/summary.csv --output runs/e0/profiling
python -m cai_lab.cli e0-cache --profile-root runs/e0/profiling --admitted-csv runs/e0/profiling/admitted_modes.csv --output runs/e0/cache
```

## 4) Carbon Data (Electricity Maps)

Option A:

```powershell
$env:ELECTRICITY_MAPS_API_KEY="<your_key>"
python -m cai_lab.cli carbon-fetch --matrix runs/frozen_matrix.yaml --output runs/carbon
```

Option B:

```powershell
python -m cai_lab.cli carbon-fetch --matrix runs/frozen_matrix.yaml --output runs/carbon --api-key-file .\secrets\electricity_maps.key
```

Option C:
- copy `configs/secrets.example.yaml` -> `configs/secrets.yaml`
- set `electricity_maps_api_key`

```powershell
python -m cai_lab.cli carbon-fetch --matrix runs/frozen_matrix.yaml --output runs/carbon
```


## NESO Carbon Conversion

If you use pre-downloaded NESO CSV bundle instead of Electricity Maps API, convert it into the standard `carbon_csv` schema first.

Matrix config:
- `configs/experiment_matrix_neso.yaml`

Converter:
- `scripts/convert_neso_bundle_to_carbon_csv.py`
- `scripts/run_neso_carbon_convert.ps1`

Example:

```powershell
python scripts/convert_neso_bundle_to_carbon_csv.py --matrix configs/experiment_matrix_neso.yaml --bundle-root E:/a-carbon/neso_carbon_intensity_bundle --output-csv runs/carbon_neso/neso_carbon.csv --output-meta runs/carbon_neso/neso_carbon_metadata.yaml --source regional
```

Then run experiments with:

```powershell
python -m cai_lab.cli e1-run --matrix runs/frozen_matrix_neso.yaml --admitted-csv runs/e0/profiling/admitted_modes.csv --cache-csv runs/e0/cache/offline_cache.csv --traces-root runs/traces_neso --carbon-csv runs/carbon_neso/neso_carbon.csv --switch-penalty-yaml runs/e0/switch_penalty/switch_penalties.yaml --load nominal --output runs/e1_neso
```

## 5) Trace Generation

```powershell
python -m cai_lab.cli trace-generate --matrix runs/frozen_matrix.yaml --output runs/traces --mu-ref vision=55 nlp=90
```

Quick dry run:

```powershell
python -m cai_lab.cli trace-generate --matrix runs/frozen_matrix.yaml --output runs/traces_smoke --mu-ref vision=0.5 nlp=1.0
```

## 6) Main Experiments

### E1

```powershell
python -m cai_lab.cli e1-run --matrix runs/frozen_matrix.yaml --admitted-csv runs/e0/profiling/admitted_modes.csv --cache-csv runs/e0/cache/offline_cache.csv --traces-root runs/traces --carbon-csv runs/carbon/electricity_maps_carbon.csv --switch-penalty-yaml runs/e0/switch_penalty/switch_penalties.yaml --load nominal --output runs/e1
```

### E2

```powershell
python -m cai_lab.cli e2-run --matrix runs/frozen_matrix.yaml --admitted-csv runs/e0/profiling/admitted_modes.csv --cache-csv runs/e0/cache/offline_cache.csv --traces-root runs/traces --carbon-csv runs/carbon/electricity_maps_carbon.csv --switch-penalty-yaml runs/e0/switch_penalty/switch_penalties.yaml --output runs/e2
```

### E3

```powershell
python -m cai_lab.cli e3-run --matrix runs/frozen_matrix.yaml --admitted-csv runs/e0/profiling/admitted_modes.csv --cache-csv runs/e0/cache/offline_cache.csv --traces-root runs/traces --carbon-csv runs/carbon/electricity_maps_carbon.csv --switch-penalty-yaml runs/e0/switch_penalty/switch_penalties.yaml --output runs/e3
```

### E4

```powershell
python -m cai_lab.cli e4-run --matrix runs/frozen_matrix.yaml --admitted-csv runs/e0/profiling/admitted_modes.csv --cache-csv runs/e0/cache/offline_cache.csv --traces-root runs/traces --carbon-csv runs/carbon/electricity_maps_carbon.csv --switch-penalty-yaml runs/e0/switch_penalty/switch_penalties.yaml --output runs/e4
```

## 7) Statistics

```powershell
python -m cai_lab.cli stats-summary --summary-csv runs/e1/e1_summary.csv --output runs/e1/e1_summary_ci95.csv
python -m cai_lab.cli stats-significance --summary-csv runs/e1/e1_summary.csv --output runs/e1/e1_significance.csv
```

## 8) Smoke Scripts

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_step1_e0.ps1 -Synthetic
powershell -ExecutionPolicy Bypass -File scripts/run_smoke_e1.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_smoke_e2.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_smoke_e3.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_smoke_e4.ps1
```

## Spike Configs

- Vision precision backend spike: `configs/spike/vision_precision_backend_spike.yaml`
- Backend readiness probe report: `runs/spike/vision_precision_backend_probe.json`

- TensorRT spike energy audit: `runs/spike/vision_tensorrt_energy_backend_audit.json`
- TensorRT spike measurement contract: `configs/measurement_contract_tensorrt_spike.yaml`


## Generated Assets Intake (2026-03-17)

Asset root:
- `E:/a-carbon/experiment_data_assets`

Ingested split files:
- `data/splits/agnews_calibration_2000.csv`
- `data/splits/agnews_profile_5000.csv`
- `data/splits/resnet50_imagenette160_proxy_calibration_1000.csv`
- `data/splits/resnet50_imagenette160_proxy_profile_3925.csv`

Spike configs:
- `configs/spike/nlp_agnews_precision_spike.yaml`
- `configs/spike/vision_imagenette_proxy_profile.yaml`

Ingest report:
- `runs/assets_ingest_report.json`
