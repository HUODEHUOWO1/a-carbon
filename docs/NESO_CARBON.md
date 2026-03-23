# NESO Carbon Workflow

This project can run E1/E2/E3/E4 with pre-downloaded NESO CSV files, without Electricity Maps API.

## Files

- Matrix: `configs/experiment_matrix_neso.yaml`
- Converter: `scripts/convert_neso_bundle_to_carbon_csv.py`
- Wrapper: `scripts/run_neso_carbon_convert.ps1`
- Bundle root: `E:/a-carbon/neso_carbon_intensity_bundle`

## Region mapping

The NESO matrix uses regional columns from `neso_regional_carbon_intensity_forecast.csv`:

- `gb_north_scotland` -> `North Scotland`
- `gb_london` -> `London`
- `gb_south_wales` -> `South Wales`

## Convert bundle to carbon CSV

```powershell
python scripts/convert_neso_bundle_to_carbon_csv.py --matrix configs/experiment_matrix_neso.yaml --bundle-root E:/a-carbon/neso_carbon_intensity_bundle --output-csv runs/carbon_neso/neso_carbon.csv --output-meta runs/carbon_neso/neso_carbon_metadata.yaml --source regional
```

## Freeze NESO matrix

```powershell
python -m cai_lab.cli freeze-matrix --matrix configs/experiment_matrix_neso.yaml --output runs/frozen_matrix_neso.yaml
```

## Run with NESO carbon

```powershell
python -m cai_lab.cli trace-generate --matrix runs/frozen_matrix_neso.yaml --output runs/traces_neso --mu-ref vision=55 nlp=90
python -m cai_lab.cli e1-run --matrix runs/frozen_matrix_neso.yaml --admitted-csv runs/e0/profiling/admitted_modes.csv --cache-csv runs/e0/cache/offline_cache.csv --traces-root runs/traces_neso --carbon-csv runs/carbon_neso/neso_carbon.csv --switch-penalty-yaml runs/e0/switch_penalty/switch_penalties.yaml --load nominal --output runs/e1_neso
```
