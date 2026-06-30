# Reproducibility Notes

## Recommended Environment

- Python 3.10+
- PyTorch 2.0+
- CUDA GPU for full experiments
- `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`, and optional `transformers`

## Seeds

The default seed is `2026`. Use:

```bash
--seed 2026 --train_ratio_seed 2026
```

For stricter deterministic behavior, add:

```bash
--deterministic 1
```

This can reduce performance and does not guarantee complete bitwise reproducibility across GPUs or PyTorch versions.

## Low-Resource Protocol

`--train_ratio` only subsamples training windows. Validation and test splits are unchanged. This is implemented in `data_provider/data_loader.py` through `split_indices` and `_apply_train_ratio`.

Available subsampling modes:

- `uniform`: evenly spaced training windows.
- `random`: seeded random subset.
- `prefix`: earliest training windows.

## Two-Stage Training

A typical paper-style run has two stages:

1. Numeric warm-up with `--method_profile numeric_only`.
2. Full RCARE training with `--method_profile privileged_bridge`, optionally loading the warm-up checkpoint through `--pretrained_numeric_checkpoint` and freezing it with `--freeze_numeric_backbone 1`.

The code also supports joint fine-tuning by leaving `--freeze_numeric_backbone 0` and setting `--numeric_learning_rate` to a small value.

## Metrics

By default, metrics are computed on standardized values. If `--inverse` is passed, evaluation uses original-scale values. Even without `--inverse`, original-scale metrics are stored in `*_inverse` keys inside `metrics.json`.

## Expected Artifacts

```text
checkpoints/<setting>/checkpoint.pth
checkpoints/<setting>/train_history.json
outputs/<setting>/metrics.json
results.txt
```

The `metrics.json` file is the most reliable artifact for tables and robustness analysis.

## Common Failure Modes

- `Text CSV not found`: generate text artifacts or update `--text_path`.
- `Text feature NPZ not found`: remove `--text_feature_path` or generate the NPZ.
- `Missing text feature column`: make sure the selected `--student_text_cols` and `--teacher_text_cols` exist in the text CSV/NPZ.
- Empty split: check `--split_mode`, dataset length, `seq_len`, and `pred_len`.
- Shape mismatch when loading a numeric checkpoint: the warm-up and full run must use compatible `seq_len`, `pred_len`, feature setting, and numeric dimensions.
