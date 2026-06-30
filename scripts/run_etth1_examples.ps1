# PowerShell example for Windows users.
# Run from the repository root after preparing dataset/ETTh1.csv and generated/ETTh1/*.csv/*.npz.
$env:PYTHON = if ($env:PYTHON) { $env:PYTHON } else { "python" }
bash scripts/run_numeric_warmup_etth1.sh
# Optionally set $env:NUMERIC_CKPT before the full RCARE run.
bash scripts/run_rcare_etth1.sh
