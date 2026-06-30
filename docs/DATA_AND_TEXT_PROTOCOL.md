# Data and Text Protocol

This document describes the data/text boundary used by RCARE-Forecast.

## Numeric Windows

For each row in a text artifact, the dataset loader reads:

- `start_idx`: first index of the historical input window.
- `pred_start_idx`: first index of the prediction horizon.
- `pred_end_idx`: last index covered by the prediction horizon.

The model consumes `X_t` from `start_idx : start_idx + seq_len` and predicts `Y_t` from `pred_start_idx : pred_start_idx + pred_len`.

## Student-Side Text

Student-side text must be available at deployment time. It can describe only the historical window or a history-derived numeric prior. Recommended columns:

- `llm_history_text`
- `llm_history_prior_text`
- `history_text`
- `compact_text`
- `paraphrase_text`

These fields may be used in `--student_text_cols`.

## Teacher-Only Text

Teacher-only text may be constructed from future windows or residuals, but it is allowed only during training:

- `llm_future_text`
- `llm_residual_text`
- `llm_privileged_text`
- `future_text`
- `residual_text`

These fields may be used in `--teacher_text_cols`. They must not be included in `--student_text_cols`.

## Unreliable Text Variants

The reliability module is trained/evaluated with corrupted history-side variants:

- `contradictory_text`
- `noisy_text`
- `missing_text`
- `time_shift_text`
- `irrelevant_text`

These variants are historical-text perturbations. They do not contain future labels.

## Text Feature Files

If `--text_feature_path` points to an NPZ file, it should contain one array per text column. Each array must have shape:

```text
[num_windows, text_dim]
```

The loader validates row count and feature dimension. If an NPZ is not provided, deterministic hash/hybrid features are built from the text CSV.

## Leakage Checks

A valid deployment configuration should satisfy:

- `--student_text_cols` contains only history-side columns.
- `--teacher_text_cols` may contain future/residual columns only during training.
- `metrics.json` reports zero deploy-equivalence differences for student prediction, numeric base, student gate, and reliability.

The implementation compares forward passes with and without teacher text during testing and stores the result in `deploy_equivalence_check`.
