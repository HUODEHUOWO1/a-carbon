# a-carbon

CAI experiment workspace for ICWS 2026 frozen execution plan.

Quick start (dry-run):

```powershell
cd E:\a-carbon
$env:PYTHONPATH="E:\a-carbon\src"
python -m cai_lab.cli freeze-matrix
python -m cai_lab.cli e0-profile --synthetic
python -m cai_lab.cli e0-prune
python -m cai_lab.cli e0-cache
python -m cai_lab.cli e0-switch-default
```

Real profiling (live GPU, no Electricity Maps required):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_real_minimal.ps1 -Config configs/profiling_real_resnet50.yaml
```

Then export real profiling outputs to precomputed root:

```powershell
python -m cai_lab.cli profile-real-export --config configs/profiling_real_resnet50.yaml --output-root runs/input_profiles
```

Carbon key sources:
- `--api-key`
- `--api-key-file`
- env `ELECTRICITY_MAPS_API_KEY`
- `configs/secrets.yaml` (`electricity_maps_api_key`)

Full workflow: `docs/EXECUTION.md`


Frozen measurement contract: `configs/measurement_contract.yaml` and `docs/MEASUREMENT_CONTRACT.md`.
Vision `capacity_k` line is archived as negative result (`runs/archives/vision_capacity_k_negative_result.yaml`) and is excluded from formal matrix.
