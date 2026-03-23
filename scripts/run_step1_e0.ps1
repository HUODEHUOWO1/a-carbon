param(
  [string]$Matrix = "configs/experiment_matrix.yaml",
  [string]$Frozen = "runs/frozen_matrix.yaml",
  [string]$PrecomputedRoot = "",
  [switch]$Synthetic,
  [switch]$StrictPrecomputed
)

$env:PYTHONPATH = "e:\a-carbon\src"

python -m cai_lab.cli freeze-matrix --matrix $Matrix --output $Frozen

if ($PrecomputedRoot -ne "") {
  if ($StrictPrecomputed) {
    python -m cai_lab.cli e0-profile --matrix $Frozen --output runs/e0/profiling --precomputed-root $PrecomputedRoot --strict-precomputed
  } else {
    python -m cai_lab.cli e0-profile --matrix $Frozen --output runs/e0/profiling --precomputed-root $PrecomputedRoot
  }
} elseif ($Synthetic) {
  python -m cai_lab.cli e0-profile --matrix $Frozen --output runs/e0/profiling --synthetic
} else {
  throw "Provide -PrecomputedRoot or -Synthetic"
}

python -m cai_lab.cli e0-prune --matrix $Frozen --profile-summary runs/e0/profiling/summary.csv --output runs/e0/profiling
python -m cai_lab.cli e0-cache --profile-root runs/e0/profiling --admitted-csv runs/e0/profiling/admitted_modes.csv --output runs/e0/cache
python -m cai_lab.cli e0-switch-default --admitted-csv runs/e0/profiling/admitted_modes.csv --output runs/e0/switch_penalty/switch_penalties.yaml