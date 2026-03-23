param(
  [string]$Matrix = "configs/experiment_matrix_neso.yaml",
  [string]$BundleRoot = "E:/a-carbon/neso_carbon_intensity_bundle",
  [string]$OutputCsv = "runs/carbon_neso/neso_carbon.csv",
  [string]$OutputMeta = "runs/carbon_neso/neso_carbon_metadata.yaml",
  [string]$Source = "regional",
  [string]$NationalMode = "prefer_actual"
)

$ErrorActionPreference = "Stop"
& "D:/Anaconda/python.exe" "scripts/convert_neso_bundle_to_carbon_csv.py" `
  --matrix $Matrix `
  --bundle-root $BundleRoot `
  --output-csv $OutputCsv `
  --output-meta $OutputMeta `
  --source $Source `
  --national-mode $NationalMode
