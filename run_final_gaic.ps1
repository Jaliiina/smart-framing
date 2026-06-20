# Final reliable runner for the smart-framing project.
# Uses GAIC-style learned candidate ranking and disables the slow local optimizer.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\cvProject\smart-framing"
& "D:\conda_envs\cv_env\python.exe" ".\eval\predict_test_b.py" `
  --image-dir "..\testB" `
  --config ".\config.gaic.yaml" `
  --output-dir ".\outputs\test_b_gaic_run"
Write-Host "Done. Output: D:\cvProject\smart-framing\outputs\test_b_gaic_run"
