param(
  [string]$Config = "configs/profiling_real_resnet50.yaml"
)

$env:PYTHONPATH = "e:\a-carbon\src"

# 1) Run live profiling for minimal real modes.
python -m cai_lab.cli profile-real-run --config $Config --mode-ids fp16_k4 int8_k4 --allow-partial-modes

# 2) Measure switch penalty for the same pair.
python -m cai_lab.cli profile-real-switch --config $Config --pairs fp16_k4:int8_k4 --n-requests 2000

# 3) Validate artifact completeness.
python -m cai_lab.cli profile-real-validate --config $Config

# 4) Export to E0 precomputed format used by e0-profile --precomputed-root.
python -m cai_lab.cli profile-real-export --config $Config --output-root runs/input_profiles