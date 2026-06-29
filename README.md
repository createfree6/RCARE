# CARE-Forecast

CARE-Forecast is a low-resource multivariate time-series forecasting framework that uses privileged semantic information during training while keeping the deployed model history-only. The numerical backbone learns the main dynamics; a privileged teacher sees future/residual semantic descriptions only during training; a deployable student distills this guidance from history-derived text and applies reliability-calibrated residual corrections at test time.

This clean repository contains the paper-facing main experiment code for the 12-dataset, 4-horizon, 3-train-ratio protocol used in our experiments.

## Main Idea

The method is designed around a strict information boundary:

- At training time, the teacher can use privileged future and residual text to expose qualitative forecast errors and correction patterns.
- At test time, the student receives only history-derived text, never future or residual text.
- Reliability calibration controls whether the semantic residual should affect the numeric forecast.
- The final prediction is a deployable numeric forecast plus a bounded, reliability-aware semantic residual.

## Repository Layout

```text
CARE-Forecast-GitHub/
  run.py                                      # single-case training/evaluation entry point
  data_provider/                              # CSV loading, splits, low-resource sampling, text features
  exp/                                        # training, validation, testing, calibration
  models/                                     # CARE-Forecast teacher/student model
  layers/                                     # RevIN and reusable layers
  utils/                                      # metrics and training utilities
  care_forecast/                              # deterministic text vectorizers
  tools/
    prepare_care_text.py                      # base text/statistical field generation
    combine_text_features.py                  # deterministic hybrid text features
    prepare_semantic_template_text_artifacts.py
    run_numeric_base_tuning_grid.py           # numeric-only tuning grid
    run_semantic_v1_lowresource_grid.py       # main semantic-v1 full grid
    export_llm_prompts.py                     # optional LLM prompt export
    generate_llm_text_openai.py               # optional OpenAI generation
    generate_llm_text_local.py                # optional local HuggingFace generation
    merge_llm_text.py                         # optional LLM output merge
  dataset/                                    # 12 CSV datasets used by the main protocol
  tables/                                     # reference targets and completed main-result tables
  scripts/                                    # clean PowerShell/Bash launchers
```

Runtime directories (`generated/`, `checkpoints/`, `outputs*`, `logs/`) are intentionally empty in the clean release and are recreated by the scripts.

## Installation

Python 3.9+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional LLM and transformer-encoder utilities require:

```bash
python -m pip install -r requirements-optional.txt
```

## Data

The clean release includes the 12 datasets used in the formal main experiment:

```text
ETTh1, ETTh2, appliances_energy, AQShunyi, AQWan, beijing_pm25,
CzeLan, exchange_rate, Flight, weather, Wind, ZafNoo
```

Default protocol:

- `seq_len = 96`
- `pred_len in {96, 192, 336, 720}`
- `train_ratio in {0.05, 0.10, 0.20}`
- `features = M`
- `target = OT`
- ETT datasets use the standard ETT split; the other datasets use chronological ratio splits.
- `train_ratio` subsamples training windows only; validation and test windows are not subsampled.

## Text Protocol

The default main experiment uses the deterministic `semantic_v1` text protocol. It creates:

- Deployable student text: `llm_history_text`, `llm_history_prior_text`
- Training-only teacher text: `llm_future_text`, `llm_residual_text`
- Robustness/noise text views for reliability checks
- Deterministic hybrid text features saved as compressed NPZ files

The accepted history template is:

```text
From [t1] to [t2] sampled every [interval], the target changed from [first] to [last] with net trend [T] ([trend_label]). Recent level is [recent_relation] earlier history, variability is [volatility_label], and recurrence is strongest at lag [L]. Related variables: [positive_vars]; opposite variables: [negative_vars]. History-only confidence: [confidence].
```

The default pipeline is fully self-contained and does not require calling an external LLM. Optional LLM scripts are included for generating alternative text fields, but the main paper-facing protocol is the deterministic `semantic_v1` pipeline.

## Quick Start: One Smoke Case

Run a small ETTh1 case first to verify the environment.

```powershell
python tools/prepare_semantic_template_text_artifacts.py --datasets ETTh1 --pred-lens 96
python tools/run_semantic_v1_lowresource_grid.py `
  --datasets ETTh1 `
  --pred-lens 96 `
  --ratios 0.10 `
  --output-dir ./outputs_semantic_v1_tunedbase `
  --result-stem smoke_etth1_semantic_v1 `
  --resume `
  --continue-on-error
```

Equivalent Bash:

```bash
python tools/prepare_semantic_template_text_artifacts.py --datasets ETTh1 --pred-lens 96
python tools/run_semantic_v1_lowresource_grid.py \
  --datasets ETTh1 \
  --pred-lens 96 \
  --ratios 0.10 \
  --output-dir ./outputs_semantic_v1_tunedbase \
  --result-stem smoke_etth1_semantic_v1 \
  --resume \
  --continue-on-error
```

## Full Main Experiment

### Step 1: Prepare semantic-v1 text artifacts

```powershell
python tools/prepare_semantic_template_text_artifacts.py `
  --datasets ETTh1,ETTh2,appliances_energy,AQShunyi,AQWan,beijing_pm25,CzeLan,exchange_rate,Flight,weather,Wind,ZafNoo `
  --pred-lens 96,192,336,720 `
  --seq-len 96 `
  --text-dim 256
```

### Step 2: Tune numeric-only base models

This step builds local numeric checkpoints and writes `tables/numeric_base_registry.csv`.

```powershell
python tools/run_numeric_base_tuning_grid.py `
  --datasets ETTh1,ETTh2,appliances_energy,AQShunyi,AQWan,beijing_pm25,CzeLan,exchange_rate,Flight,weather,Wind,ZafNoo `
  --pred-lens 96,192,336,720 `
  --ratios 0.05,0.10,0.20 `
  --seq-len 96 `
  --target-csv tables/excel_numeric_baseline_targets.csv `
  --target-tolerance 1.10 `
  --max-candidates 6 `
  --only-bad `
  --resume
```

### Step 3: Run semantic-v1 tuned-base grid

```powershell
python tools/run_semantic_v1_lowresource_grid.py `
  --datasets ETTh1,ETTh2,appliances_energy,AQShunyi,AQWan,beijing_pm25,CzeLan,exchange_rate,Flight,weather,Wind,ZafNoo `
  --pred-lens 96,192,336,720 `
  --ratios 0.05,0.10,0.20 `
  --seq-len 96 `
  --numeric-registry tables/numeric_base_registry.csv `
  --output-dir ./outputs_semantic_v1_tunedbase `
  --result-stem semantic_v1_tunedbase_lowresource_results `
  --resume `
  --continue-on-error
```

You can also run the same sequence with the launchers:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_prepare_semantic_v1.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_numeric_tuning.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_semantic_v1_tunedbase_grid.ps1
```

Or one command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_main_experiment.ps1
```

Bash launchers are also provided:

```bash
bash scripts/run_prepare_semantic_v1.sh
bash scripts/run_numeric_tuning.sh
bash scripts/run_semantic_v1_tunedbase_grid.sh
# or:
bash scripts/run_full_main_experiment.sh
```

## Outputs

Main outputs are written to:

```text
tables/semantic_v1_tunedbase_lowresource_results.csv
tables/semantic_v1_tunedbase_lowresource_results.md
tables/semantic_v1_tunedbase_lowresource_results_dataset_summary.csv
tables/semantic_v1_tunedbase_lowresource_results_ratio_horizon_summary.csv
outputs_semantic_v1_tunedbase/semantic_v1_tunedbase_lowresource_results.json
```

Numeric tuning outputs are written to:

```text
tables/numeric_base_registry.csv
tables/numeric_base_registry.md
outputs_numeric_tuned/numeric_base_tuning_results.json
```

Reference result tables from the completed local run are included under `tables/`. Checkpoints and generated feature files are not included because they are large and reproducible.

## Single-Case Training

For manual experiments, call `run.py` directly. Example numeric-only run:

```powershell
python run.py `
  --is_training 1 `
  --model CARE_Forecast `
  --data ETTh1 `
  --root_path . `
  --data_path dataset/ETTh1.csv `
  --text_path generated/ETTh1/ETTh1_sl96_pl96_text_M_semantic_v1.csv `
  --text_feature_path generated/ETTh1/ETTh1_sl96_pl96_semantic_v1_text_features.npz `
  --features M `
  --target OT `
  --split_mode ett_standard `
  --seq_len 96 `
  --label_len 48 `
  --pred_len 96 `
  --method_profile numeric_only `
  --ablation numeric_only `
  --train_ratio 0.10 `
  --train_ratio_seed 2026 `
  --train_ratio_mode uniform
```

The main semantic run is usually easier to launch through `tools/run_semantic_v1_lowresource_grid.py`, because it handles artifact preparation, numeric checkpoint reuse, batch-size choices, residual calibration, and table export.

## Optional LLM Text Generation

The main results do not require an external LLM. If you want to replace deterministic semantic fields with LLM-generated fields:

```powershell
python tools/export_llm_prompts.py --text-csv generated/ETTh1/ETTh1_sl96_pl96_text_M.csv --output-jsonl generated/ETTh1/ETTh1_sl96_pl96_llm_prompts.jsonl
$env:OPENAI_API_KEY = "YOUR_API_KEY"
python tools/generate_llm_text_openai.py --prompts-jsonl generated/ETTh1/ETTh1_sl96_pl96_llm_prompts.jsonl --output-jsonl generated/ETTh1/ETTh1_sl96_pl96_openai_outputs.jsonl --resume
python tools/merge_llm_text.py --base-csv generated/ETTh1/ETTh1_sl96_pl96_text_M.csv --outputs-jsonl generated/ETTh1/ETTh1_sl96_pl96_openai_outputs.jsonl --output-csv generated/ETTh1/ETTh1_sl96_pl96_text_M_llm.csv
```

For local HuggingFace models, set `HF_HUB_CACHE` or pass `--hub-dir` to `tools/generate_llm_text_local.py`.

## Reproducibility Notes

- Use the same `--seed 2026` for paper-facing runs unless you are doing multi-seed analysis.
- Use `--resume` for long grids; completed cases are reused.
- Use `--continue-on-error` for full grids so one failed case does not stop the whole run.
- If CUDA memory is limited, reduce `--force-batch-size` and `--force-eval-batch-size` in `tools/run_semantic_v1_lowresource_grid.py`.
- Check `tables/*_failures.csv` after any full run.

## Citation

A formal citation will be added after the manuscript is public.
