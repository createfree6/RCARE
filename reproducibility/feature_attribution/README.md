# Feature-attribution records

This directory contains the released records for the deterministic feature-attribution control reported in Table 10. The control compares the reported `Hybrid` input with a `Numeric summary` input and a `Template hash` input.

## Protocol

- Datasets: `AQShunyi` and `weather`
- Prediction horizon: `H=96`
- Training windows: `10%`
- Fixed train-window subset seed: `2026`
- Initialization seeds: `2026`, `2027`, and `2028`
- Total records: 18

The numeric-summary input retains only positions 24-40 and 48-58 of each 256-dimensional field. It excludes template-phrase flags and the 160-dimensional hash. The template-hash input retains positions 96-255 only. Each field is L2-normalized, and the student and teacher each concatenate two fields into a 512-dimensional input.

## Files

- `feature_attribution_h96_train10_runs.csv`: run-level metrics for all 18 runs.
- `feature_attribution_h96_train10_summary.csv`: means and sample standard deviations over three initialization seeds.
- `feature_attribution_h96_train10_receipt.json`: consolidated protocol receipt and run metadata.
- `verify_records.py`: recomputes each reported mean and sample standard deviation from the run-level CSV and checks the receipt.

The path fields inside the released run records retain their original execution-directory labels, including `r1q2`. They are historical provenance fields rather than public filenames or repository paths.

Run the standalone audit with:

```bash
python reproducibility/feature_attribution/verify_records.py
```

## Re-running the control

`tools/run_feature_attribution.py` reconstructs the matched feature controls and launches the experiment protocol. It requires the public raw dataset, the cached `semantic_v1` text CSV and feature files, and the numeric-prior checkpoints prepared by the base training protocol. Those large inputs are not included in this records directory. Once they are prepared as described in `docs/DATA_AND_TEXT_PROTOCOL.md`, run:

```bash
python tools/run_feature_attribution.py --datasets AQShunyi,weather --pred-len 96 --train-ratio 0.10 --seeds 2026,2027,2028 --train-ratio-seed 2026
```
