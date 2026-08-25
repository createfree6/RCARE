"""Verify the released run-level records and their aggregate statistics."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "feature_attribution_h96_train10_runs.csv"
SUMMARY = ROOT / "feature_attribution_h96_train10_summary.csv"
RECEIPT = ROOT / "feature_attribution_h96_train10_receipt.json"
METRICS = (
    "student_mse",
    "student_mae",
    "teacher_oracle_mse",
    "teacher_oracle_mae",
    "mean_gate",
    "mean_reliability",
)
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return abs(left - right) <= TOLERANCE


def main() -> None:
    with RUNS.open(newline="", encoding="utf-8-sig") as handle:
        runs = list(csv.DictReader(handle))
    with SUMMARY.open(newline="", encoding="utf-8-sig") as handle:
        summary_rows = list(csv.DictReader(handle))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))

    if len(runs) != 18:
        raise ValueError(f"Expected 18 run records, found {len(runs)}.")
    if len(receipt["records"]) != len(runs):
        raise ValueError("The receipt record count does not match the run-level CSV.")
    if receipt["datasets"] != ["AQShunyi", "weather"]:
        raise ValueError(f"Unexpected datasets in receipt: {receipt['datasets']}")
    if receipt["pred_len"] != 96 or receipt["train_ratio"] != 0.10:
        raise ValueError("The receipt does not describe the H=96, 10% protocol.")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        grouped[(row["dataset"], row["variant"])].append(row)
    if len(grouped) != 6 or any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("Expected three initialization runs for each dataset-variant pair.")

    summaries = {(row["dataset"], row["variant"]): row for row in summary_rows}
    if set(summaries) != set(grouped):
        raise ValueError("The summary rows do not match the run-level dataset-variant pairs.")
    for key, rows in grouped.items():
        summary = summaries[key]
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            if not close(statistics.mean(values), float(summary[f"{metric}_mean"])):
                raise ValueError(f"Mean mismatch for {key}, {metric}.")
            if not close(statistics.stdev(values), float(summary[f"{metric}_std"])):
                raise ValueError(f"Standard deviation mismatch for {key}, {metric}.")

    print("Feature-attribution records verified.")
    for path in (RUNS, SUMMARY, RECEIPT):
        print(f"{path.name}: {sha256(path)}")


if __name__ == "__main__":
    main()
