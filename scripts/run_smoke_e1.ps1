param(
  [string]$Matrix = "runs/frozen_matrix.yaml"
)

$env:PYTHONPATH = "e:\a-carbon\src"

python -m cai_lab.cli trace-generate --matrix $Matrix --output runs/traces_smoke --mu-ref vision=0.5 nlp=1.0
python -m cai_lab.cli e1-run --matrix $Matrix --admitted-csv runs/e0/profiling/admitted_modes.csv --cache-csv runs/e0/cache/offline_cache.csv --traces-root runs/traces_smoke --carbon-csv runs/carbon/electricity_maps_carbon.csv --switch-penalty-yaml runs/e0/switch_penalty/switch_penalties.yaml --load nominal --controllers static_hq reactive_joint forecast_budgeted_joint --regions eu --seasons summer_week --max-traces-per-workload 1 --max-requests-per-trace 200 --output runs/e1_smoke