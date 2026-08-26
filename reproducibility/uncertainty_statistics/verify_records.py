"""Verify the released Reviewer 1 Q3 uncertainty and statistical records."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MULTISEED_RUNS = ROOT / "r1q3_multiseed_fixedsubset_runs.csv"
MULTISEED_SUMMARY = ROOT / "r1q3_multiseed_fixedsubset_summary.csv"
MULTISEED_RECEIPT = ROOT / "r1q3_multiseed_fixedsubset_receipt.json"
WILCOXON = ROOT / "r1q3_wilcoxon_full.csv"
WILCOXON_RECEIPT = ROOT / "r1q3_wilcoxon_full_receipt.json"
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    return abs(left - right) <= TOLERANCE


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return adjusted q-values in the original input order."""
    total = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * total
    previous = 1.0
    for rank in range(total, 0, -1):
        index, p_value = ordered[rank - 1]
        previous = min(previous, p_value * total / rank)
        adjusted[index] = min(previous, 1.0)
    return adjusted


def verify_multiseed() -> None:
    with MULTISEED_RUNS.open(newline="", encoding="utf-8-sig") as handle:
        runs = list(csv.DictReader(handle))
    with MULTISEED_SUMMARY.open(newline="", encoding="utf-8-sig") as handle:
        summaries = list(csv.DictReader(handle))
    receipt = json.loads(MULTISEED_RECEIPT.read_text(encoding="utf-8"))

    protocol = receipt["protocol"]
    if protocol["initialization_seeds"] != [2026, 2027, 2028]:
        raise ValueError("Unexpected initialization seeds in the multi-seed receipt.")
    if protocol["train_ratio_seed"] != 2026:
        raise ValueError("Unexpected train-window selection seed in the receipt.")
    if len(runs) != 18 or len(receipt["records"]) != 18:
        raise ValueError("The multi-seed record count should be 18.")

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        key = (row["role"], row["dataset"], row["pred_len"], row["method"])
        grouped[key].append(row)
    if len(grouped) != 6 or any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("Expected three rows for each of six dataset-method entries.")

    summary_by_key = {
        (row["role"], row["dataset"], row["pred_len"], row["method"]): row
        for row in summaries
    }
    if set(summary_by_key) != set(grouped):
        raise ValueError("The multi-seed summary does not match the run-level entries.")

    for key, rows in grouped.items():
        summary = summary_by_key[key]
        if int(summary["runs"]) != len(rows):
            raise ValueError(f"Incorrect run count in summary for {key}.")
        for metric in ("mse", "mae"):
            values = [float(row[metric]) for row in rows]
            if not close(statistics.mean(values), float(summary[f"{metric}_mean"])):
                raise ValueError(f"Mean mismatch for {key}, {metric}.")
            if not close(statistics.stdev(values), float(summary[f"{metric}_std"])):
                raise ValueError(f"Sample standard deviation mismatch for {key}, {metric}.")


def verify_wilcoxon() -> None:
    with WILCOXON.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    receipt = json.loads(WILCOXON_RECEIPT.read_text(encoding="utf-8"))

    if len(rows) != 66 or receipt["rows"] != 66:
        raise ValueError("The Wilcoxon record count should be 66.")
    if receipt["paired_datasets"] != [12]:
        raise ValueError("The Wilcoxon receipt should use 12 paired datasets.")

    families: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["n"]) != 12:
            raise ValueError("Every Wilcoxon comparison should use 12 paired datasets.")
        families[(row["ratio"], row["metric"])].append(row)
    if len(families) != 6 or any(len(rows) != 11 for rows in families.values()):
        raise ValueError("Expected six correction families with 11 baselines each.")

    for family, family_rows in families.items():
        q_values = benjamini_hochberg(
            [float(row["wilcoxon_p_two_sided"]) for row in family_rows]
        )
        for row, q_value in zip(family_rows, q_values):
            reported = float(row["BH_FDR_q_two_sided"])
            if not close(q_value, reported):
                raise ValueError(f"BH q-value mismatch for {family}, {row['baseline']}.")


def main() -> None:
    verify_multiseed()
    verify_wilcoxon()
    print("Uncertainty and Wilcoxon records verified.")
    for path in (
        MULTISEED_RUNS,
        MULTISEED_SUMMARY,
        MULTISEED_RECEIPT,
        WILCOXON,
        WILCOXON_RECEIPT,
    ):
        print(f"{path.name}: {sha256(path)}")


if __name__ == "__main__":
    main()
