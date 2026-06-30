# RCARE-Forecast

### Reliability-Calibrated Privileged Residual Learning for data-efficient text-numeric time series forecasting.

RCARE-Forecast is a PyTorch implementation of a deployable history-only forecasting model that learns from training-time privileged semantic hindsight. During training, a teacher branch can read future-pattern and residual-explanation text. During inference, only historical numeric windows and history-side text are used by the student branch. The final prediction is produced as a reliability-calibrated semantic residual correction on top of a stable numeric forecast prior.

## Highlights

- **MA-FRFT numeric prior**: moving-average decomposition with FFT/FRFT spectral modeling provides a stable numeric forecast backbone.
- **Privileged semantic teacher**: future-pattern and residual-explanation text supervise residual correction only during training.
- **History-only student**: the deployable model consumes only historical numeric data and history-side text at test time.
- **Selective residual distillation**: the student learns teacher corrections mainly when the teacher improves over the numeric prior.
- **Reliability-calibrated gating**: unreliable, missing, contradictory, or irrelevant history text is suppressed so the model can fall back toward the numeric prior.

## Repository Structure

```text
CARE-ForecastGitHub/
  run.py                         # Main training/evaluation entrypoint
  care_forecast/                 # Deterministic text feature encoders
  data_provider/                 # Sliding-window dataset and split protocol
  exp/                           # Training, validation, evaluation loops
  layers/                        # Shared layers such as RevIN
  models/                        # RCARE/CARE model implementations
  utils/                         # Metrics, early stopping, learning-rate utilities
  tools/                         # Text construction, feature export, audits, grid runners
  scripts/                       # Minimal runnable examples
  docs/                          # Protocol and reproducibility notes
  dataset/                       # Put raw time-series CSV files here (ignored by git)
  generated/                     # Put cached text CSV/NPZ artifacts here (ignored by git)
  checkpoints/                   # Training checkpoints (ignored by git)
  outputs/                       # Metrics JSON outputs (ignored by git)
```

Large datasets, generated text files, checkpoints, logs, and manuscript-specific results are intentionally excluded from this clean release.

## Installation

The code requires Python 3.10+ and PyTorch. A CUDA-enabled PyTorch installation is recommended for full experiments.

```bash
conda create -n rcare python=3.10 -y
conda activate rcare
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install it from the official PyTorch selector first, then install the remaining packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

For CPU-only debugging, install the CPU PyTorch wheel and keep the rest unchanged.

## Data Format

Place each multivariate time-series CSV under `dataset/`. The expected format follows common long-term forecasting benchmarks:

```text
date,OT,var_1,var_2,...
2016-07-01 00:00:00,30.531, ...
```

Requirements:

- The timestamp column should be named `date`.
- All non-`date` columns are treated as numeric variables.
- `--target` selects the target channel for univariate mode and the target index used by text construction.
- With `--features M`, all variables are both inputs and prediction targets.
- With `--features S`, only the target channel is predicted.

The loader supports two split protocols:

- `--split_mode ett_standard`: 12/4/4 months for ETT-style datasets.
- `--split_mode ratio`: chronological 7:1:2 split for non-ETT datasets.

Low-resource experiments use `--train_ratio` to subsample only the training windows. Validation and test windows remain complete.

## Text Artifact Protocol

RCARE-Forecast reads a cached text CSV for each `(dataset, seq_len, pred_len, feature setting)` pair. The text CSV must contain sliding-window indices and text fields:

```text
start_idx,pred_start_idx,pred_end_idx,
history_text,compact_text,paraphrase_text,
contradictory_text,noisy_text,missing_text,time_shift_text,irrelevant_text,
llm_history_text,llm_history_prior_text,llm_future_text,llm_residual_text,llm_privileged_text,
history_summary,future_summary,residual_summary,...
```

The inference boundary is implemented through column selection:

- Student-side deployable text: `--student_text_cols`, typically `llm_history_text llm_history_prior_text`.
- Teacher-only privileged text: `--teacher_text_cols`, typically `llm_future_text llm_residual_text`.
- At test time, the student prediction path can be evaluated without teacher text; `deploy_equivalence_check` verifies that student/base/gate/reliability are unchanged.

### Option A: Deterministic Template Text

For quick reproduction without calling an LLM, generate semantic template text and deterministic hybrid features:

```bash
python tools/prepare_semantic_template_text_artifacts.py \
  --datasets ETTh1 \
  --pred-lens 96 \
  --seq-len 96 \
  --target OT \
  --text-dim 256
```

This creates files such as:

```text
generated/ETTh1/ETTh1_sl96_pl96_text_M_semantic_v1.csv
generated/ETTh1/ETTh1_sl96_pl96_semantic_v1_text_features.npz
```

### Option B: Basic CARE Text Fields

For a simpler single-file artifact:

```bash
python tools/prepare_care_text.py \
  --input dataset/ETTh1.csv \
  --output generated/ETTh1_sl96_pl96_text.csv \
  --seq-len 96 \
  --pred-len 96 \
  --target OT
```

If `--text_feature_path` is not provided during training, the loader builds deterministic hash/hybrid features directly from the text CSV.

### Optional Transformer Text Features

If you want offline Transformer embeddings, use:

```bash
python tools/build_transformer_text_features.py \
  --text_csv generated/ETTh1/ETTh1_sl96_pl96_text_M_semantic_v1.csv \
  --output generated/ETTh1/ETTh1_sl96_pl96_transformer_features.npz \
  --model_name sentence-transformers/all-MiniLM-L6-v2
```

Then concatenate them with hybrid features if needed:

```bash
python tools/combine_text_features.py \
  --text_csv generated/ETTh1/ETTh1_sl96_pl96_text_M_semantic_v1.csv \
  --npz generated/ETTh1/ETTh1_sl96_pl96_transformer_features.npz \
  --include_hybrid \
  --output generated/ETTh1/ETTh1_sl96_pl96_combined_features.npz
```

## Quick Start

```bash
mkdir -p dataset generated/ETTh1
# Copy ETTh1.csv into dataset/ETTh1.csv
python tools/prepare_semantic_template_text_artifacts.py --datasets ETTh1 --pred-lens 96 --seq-len 96 --target OT
bash scripts/run_numeric_warmup_etth1.sh
bash scripts/run_rcare_etth1.sh
```

To reuse a warm-up numeric checkpoint:

```bash
NUMERIC_CKPT="checkpoints/<numeric-setting>/checkpoint.pth" bash scripts/run_rcare_etth1.sh
```

## Manual Training Command

```bash
python run.py \
  --is_training 1 \
  --model_id ETTh1_sl96_pl96_r10_rcare \
  --model CARE_Forecast \
  --method_profile privileged_bridge \
  --data ETTh1 \
  --root_path . \
  --data_path dataset/ETTh1.csv \
  --text_path generated/ETTh1/ETTh1_sl96_pl96_text_M_semantic_v1.csv \
  --text_feature_path generated/ETTh1/ETTh1_sl96_pl96_semantic_v1_text_features.npz \
  --features M \
  --target OT \
  --freq h \
  --split_mode ett_standard \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --text_dim 256 \
  --student_text_cols llm_history_text llm_history_prior_text \
  --teacher_text_cols llm_future_text llm_residual_text \
  --hidden_dim 256 \
  --sem_dim 128 \
  --moving_avg_kernel 7 \
  --frft_init_alpha 0.4 \
  --residual_planner_type cross_attn \
  --train_ratio 0.10 \
  --train_epochs 12 \
  --batch_size 32 \
  --eval_batch_size 64 \
  --learning_rate 5e-4 \
  --numeric_learning_rate 5e-5 \
  --weight_decay 1e-4 \
  --patience 5 \
  --checkpoints checkpoints \
  --output_dir outputs
```

## Important Arguments

| Argument | Meaning |
| --- | --- |
| `--method_profile numeric_only` | Train only the MA-FRFT numeric prior. |
| `--method_profile privileged_bridge` | Recommended teacher-student RCARE profile. |
| `--method_profile semantic_planner` | Enables horizon-level cross-attention residual planning. |
| `--student_text_cols` | History-only text fields available during deployment. |
| `--teacher_text_cols` | Future/residual text fields used only by the training teacher. |
| `--text_feature_path` | Optional NPZ with precomputed features for each text column. |
| `--train_ratio` | Low-resource training-window fraction. |
| `--split_mode` | Chronological split protocol (`ett_standard` or `ratio`). |
| `--moving_avg_kernel` | Moving-average kernel for trend decomposition. |
| `--frft_init_alpha` | Fractional order used by the FRFT branch. |
| `--residual_budget_type` | Residual magnitude constraint, e.g. `history_std`. |
| `--calibrate_residual` | Optional validation-time scalar residual calibration. |

## Outputs

Each run writes:

```text
checkpoints/<setting>/checkpoint.pth
checkpoints/<setting>/train_history.json
outputs/<setting>/metrics.json
results.txt
```

`metrics.json` contains clean-test metrics, unreliable-text robustness metrics, numeric-backbone information, and `deploy_equivalence_check` for no-leakage verification.

## No-Leakage Evaluation

RCARE-Forecast is designed so future/residual text is not a test-time input to the student. The evaluation loop compares two forward passes: one with teacher text attached and one with teacher text removed. The maximum absolute differences for student prediction, numeric base, student gate, and reliability should be `0.0`.

## GitHub Release Notes

- Do not commit raw datasets, generated text artifacts, checkpoints, or output metrics unless they are intentionally small examples.
- Add a license file before public release if the repository should be open-source.
- Add citation information after the manuscript is accepted or posted.
- If using LLM-generated text, document the model, prompt, decoding settings, and caching rules.

## Citation

```bibtex
@article{rcareforecast2026,
  title={Reliability-Calibrated Privileged Residual Learning for Data-Efficient Text-Numeric Time Series Forecasting},
  author={Zhao, Chunna and others},
  journal={Manuscript under review},
  year={2026}
}
```
