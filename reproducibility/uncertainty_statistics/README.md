# Uncertainty and statistical records

This directory contains the records supporting the uncertainty and statistical
analysis added in response to Reviewer 1, Question 3.

## Initialization-variability protocol

The three-seed analysis contains two representative settings:

- Strong-gain setting: `AQShunyi`, horizon `96`, and `10%` training windows.
- Weak/fallback setting: `Flight`, horizon `336`, and `20%` training windows.
- Initialization seeds: `2026`, `2027`, and `2028`.
- Within each implementation, uniform train-window selection uses the fixed
  seed `2026`. Therefore, the recorded variation is due to model
  initialization and training stochasticity, not a newly sampled window set.
- RCARE-Forecast and TimeMixer are retrained for each initialization seed.
  The MA-FRFT numeric prior is a frozen reference. Its value is repeated in
  the run-level record only to align the rows with the three-seed protocol. It
  is not three independent trainings and is labelled `fixed` in the manuscript.

The selected settings intentionally include both a stable strong-gain case and
a mixed case. These three-run summaries describe initialization variability;
they are not used to make a new statistical-significance claim. Sensitivity to
different selected training windows is a separate protocol question.

## Cross-dataset Wilcoxon protocol

The Wilcoxon file contains all 66 comparisons from the main experiment. Each
two-sided signed-rank test uses the 12 datasets as paired units. For each
training ratio and metric, Benjamini-Hochberg correction is applied over the
11 baseline comparisons. The file reports the raw p-value and the corrected
q-value for every comparison.

## Files

- `r1q3_multiseed_fixedsubset_runs.csv`: 18 run-level records.
- `r1q3_multiseed_fixedsubset_summary.csv`: sample means and sample standard
  deviations for the six dataset-method entries.
- `r1q3_multiseed_fixedsubset_receipt.json`: protocol details and consolidated
  records.
- `r1q3_wilcoxon_full.csv`: all raw p-values and corrected q-values.
- `r1q3_wilcoxon_full_receipt.json`: Wilcoxon test and correction metadata.
- `verify_records.py`: standard-library audit of the released records.

## Verification

Run the audit from the repository root:

```bash
python reproducibility/uncertainty_statistics/verify_records.py
```

The script recomputes the reported multi-seed means and sample standard
deviations, checks the multi-seed receipt, and recomputes every
Benjamini-Hochberg q-value from the Wilcoxon p-values.

## Re-running the experiments

The released records are sufficient to audit the reported aggregation and
multiple-comparison correction. Full training reruns additionally require the
original public datasets, cached structured text fields, and numeric-prior
checkpoints. Those large files are intentionally not included in this release.
