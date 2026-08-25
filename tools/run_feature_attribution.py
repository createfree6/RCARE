from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_final_paper_table_protocol as final_protocol


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
TEXT_COLUMNS = [
    "llm_history_text",
    "llm_history_prior_text",
    "llm_future_text",
    "llm_residual_text",
    "llm_privileged_text",
    "history_text",
    "compact_text",
    "paraphrase_text",
    "contradictory_text",
    "noisy_text",
    "missing_text",
    "time_shift_text",
    "irrelevant_text",
]
SUMMARY_FOR_TEXT = {
    "llm_history_text": "history_summary",
    "llm_history_prior_text": "history_prior_summary",
    "llm_future_text": "future_summary",
    "llm_residual_text": "residual_summary",
    "llm_privileged_text": "residual_summary",
    "history_text": "history_summary",
    "compact_text": "history_summary",
    "paraphrase_text": "history_summary",
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_summary(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-6)).astype(np.float32)


def hash_vector(text: str, dim: int = 160) -> np.ndarray:
    """Reproduce the deterministic signed token hash used by semantic_v1."""
    if not isinstance(text, str) or not text.strip():
        text = "No textual information available."
    import hashlib as _hashlib
    import re

    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        value = int.from_bytes(_hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
        index = value % dim
        vector[index] += -1.0 if ((value >> 8) & 1) else 1.0
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def set_summary_features(vector: np.ndarray, summary: dict[str, Any] | None) -> None:
    """Write only numerical-summary features, deliberately excluding template phrases."""
    if not summary:
        return
    categorical = {
        "trend": {"upward": 24, "downward": 25, "stable": 26},
        "volatility": {"high": 27, "medium": 28, "low": 29},
        "periodicity": {"strong": 30, "medium": 31, "weak": 32},
        "recent_change": {"recent increase": 33, "recent decrease": 34, "little recent change": 35},
        "anomaly": {"yes": 36, "no": 37},
        "uncertainty": {"high": 38, "medium": 39, "low": 40},
    }
    for key, mapping in categorical.items():
        index = mapping.get(str(summary.get(key, "")))
        if index is not None:
            vector[index] = 1.0

    numeric_keys = [
        "trend_score",
        "recent_change_score",
        "volatility_score",
        "period_corr",
        "anomaly_count",
        "first",
        "last",
        "mean",
        "std",
        "min",
        "max",
    ]
    for offset, key in enumerate(numeric_keys):
        value = safe_float(summary.get(key), 0.0)
        scale = 1.0 if key.endswith("score") or key in {"volatility_score", "period_corr", "anomaly_count"} else 10.0
        vector[48 + offset] = np.tanh(value / scale)


def build_control_features(text_csv: Path, source_npz: Path, output_dir: Path) -> dict[str, Path]:
    """Create matched 256-D inputs that isolate summary and template-hash blocks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    text_df = pd.read_csv(text_csv)
    with np.load(source_npz) as source:
        hybrid = {key: source[key].astype(np.float32) for key in source.files if key in TEXT_COLUMNS}

    expected_rows = len(text_df)
    for column in TEXT_COLUMNS:
        if column not in hybrid:
            raise ValueError(f"Missing {column} in {source_npz}")
        if hybrid[column].shape != (expected_rows, 256):
            raise ValueError(f"Unexpected shape for {column}: {hybrid[column].shape}")

    summary_only: dict[str, np.ndarray] = {}
    template_hash_only: dict[str, np.ndarray] = {}
    for column in TEXT_COLUMNS:
        structured = np.zeros((expected_rows, 256), dtype=np.float32)
        summary_column = SUMMARY_FOR_TEXT.get(column)
        summaries = text_df[summary_column].tolist() if summary_column in text_df.columns else [None] * expected_rows
        for row_index, summary_value in enumerate(summaries):
            set_summary_features(structured[row_index, :96], parse_summary(summary_value))
        summary_only[column] = normalize_rows(structured)

        hash_only = np.zeros((expected_rows, 256), dtype=np.float32)
        texts = text_df[column].fillna("No textual information available.").astype(str).tolist()
        hash_only[:, 96:] = np.stack([hash_vector(text) for text in texts], axis=0)
        template_hash_only[column] = normalize_rows(hash_only)

    paths = {
        "hybrid": source_npz,
        "structured_summary_only": output_dir / f"{text_csv.stem}_structured_summary_only_256d.npz",
        "template_hash_only": output_dir / f"{text_csv.stem}_template_hash_only_256d.npz",
    }
    np.savez_compressed(paths["structured_summary_only"], **summary_only)
    np.savez_compressed(paths["template_hash_only"], **template_hash_only)

    manifest = {
        "protocol": "semantic_v1 feature-attribution control",
        "source_text_csv": rel(text_csv),
        "source_text_csv_sha256": sha256(text_csv),
        "source_hybrid_npz": rel(source_npz),
        "source_hybrid_npz_sha256": sha256(source_npz),
        "field_dimension": 256,
        "student_input_dimension": 512,
        "teacher_input_dimension": 512,
        "variants": {
            "hybrid": "Original semantic_v1 cached vectors: 96-D semantic block plus 160-D signed template hash, L2 normalized.",
            "structured_summary_only": "Only the directly parsed structured summaries at indices 24-40 and 48-58 of a 96-D semantic block. No template text or hash block is read. Remaining positions are zero and the 256-D vector is L2 normalized.",
            "template_hash_only": "Only the original 160-D signed deterministic hash of the fixed template text at indices 96-255. The first 96 positions are zero and the 256-D vector is L2 normalized.",
        },
        "row_count": expected_rows,
        "feature_norm_range": {
            label: {
                column: [float(np.linalg.norm(values, axis=1).min()), float(np.linalg.norm(values, axis=1).max())]
                for column, values in arrays.items()
            }
            for label, arrays in {
                "hybrid": hybrid,
                "structured_summary_only": summary_only,
                "template_hash_only": template_hash_only,
            }.items()
        },
    }
    (output_dir / f"{text_csv.stem}_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return paths


def replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError:
        command.extend([option, value])
        return
    if index + 1 >= len(command):
        raise ValueError(f"Malformed command: {option} has no value")
    command[index + 1] = value


def clean_metrics(metric_path: Path) -> dict[str, float]:
    metric_data = json.loads(metric_path.read_text(encoding="utf-8"))
    clean = metric_data["test_clean"]
    return {
        "student_mse": float(clean["student_mse"]),
        "student_mae": float(clean["student_mae"]),
        "teacher_oracle_mse": float(clean["teacher_oracle_mse"]),
        "teacher_oracle_mae": float(clean["teacher_oracle_mae"]),
        "numeric_base_mse": float(clean["numeric_base_mse"]),
        "numeric_base_mae": float(clean["numeric_base_mae"]),
        "mean_gate": float(clean["mean_gate"]),
        "mean_reliability": float(clean["mean_reliability"]),
    }


def run_command(command: list[str], log_path: Path, dry_run: bool) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        log_path.write_text("DRY RUN\n$ " + " ".join(command) + "\n", encoding="utf-8")
        return 0.0
    environment = os.environ.copy()
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment.setdefault("PYTHONUTF8", "1")
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}; inspect {log_path}")
    return elapsed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["variant"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (dataset, variant), records in sorted(groups.items()):
        result: dict[str, Any] = {"dataset": dataset, "variant": variant, "runs": len(records)}
        for metric in ["student_mse", "student_mae", "teacher_oracle_mse", "teacher_oracle_mae", "mean_gate", "mean_reliability"]:
            values = np.asarray([float(record[metric]) for record in records], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(result)
    return summary_rows


def source_metric_path(ledger: pd.DataFrame, dataset: str, pred_len: int, train_ratio: float) -> Path:
    selected = ledger[
        (ledger["dataset"].astype(str) == dataset)
        & (ledger["pred_len"].astype(int) == pred_len)
        & (np.isclose(ledger["train_ratio"].astype(float), train_ratio))
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one final result row for {dataset}, H={pred_len}, ratio={train_ratio}; got {len(selected)}")
    path = ROOT / str(selected.iloc[0]["full_metric_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the semantic_v1 feature-attribution controls.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", default="AQShunyi,weather")
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.10)
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--train-ratio-seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs_feature_attribution")
    parser.add_argument("--feature-dir", default="generated/feature_attribution")
    parser.add_argument("--result-stem", default="feature_attribution_h96_train10")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not datasets or not seeds:
        raise ValueError("At least one dataset and seed are required.")

    ledger = pd.read_csv(TABLES / "semantic_v1_tunedbase_lowresource_results.csv")
    run_csv = TABLES / f"{args.result_stem}_runs.csv"
    records_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    if args.resume and run_csv.exists():
        for row in pd.read_csv(run_csv).to_dict(orient="records"):
            key = (str(row["dataset"]), str(row["variant"]), int(row["seed"]))
            records_by_key[key] = row
    failures: list[dict[str, Any]] = []
    output_root = ROOT / args.output_dir
    log_dir = output_root / "logs"

    for dataset in datasets:
        source_metric = source_metric_path(ledger, dataset, args.pred_len, args.train_ratio)
        source = json.loads(source_metric.read_text(encoding="utf-8"))
        source_args = source["args"]
        text_path = ROOT / str(source_args["text_path"])
        source_npz = ROOT / str(source_args["text_feature_path"])
        feature_paths = build_control_features(text_path, source_npz, ROOT / args.feature_dir / dataset)

        for variant, feature_path in feature_paths.items():
            for seed in seeds:
                command = final_protocol.command_from_metric(source, args.python)
                model_id = f"{dataset.lower()}_feature_attribution_{variant}_p{args.pred_len}_r{args.train_ratio:.2f}_s{seed}".replace(".", "p")
                description = f"feature_attribution_{variant}_p{args.pred_len}"
                replace_option(command, "--model_id", model_id)
                replace_option(command, "--des", description)
                replace_option(command, "--output_dir", args.output_dir)
                replace_option(command, "--text_feature_path", rel(feature_path))
                replace_option(command, "--seed", str(seed))
                replace_option(command, "--train_ratio_seed", str(args.train_ratio_seed))
                replace_option(command, "--gpu", str(args.gpu))
                setting = f"long_term_forecast_{model_id}_CARE_Forecast_{dataset}_ftM_sl96_pl{args.pred_len}_hd{source_args['hidden_dim']}_sd{source_args['sem_dim']}_{description}_0"
                metric_path = output_root / setting / "metrics.json"
                log_path = log_dir / f"{dataset}_{variant}_s{seed}.log"
                try:
                    if args.resume and metric_path.exists():
                        elapsed = 0.0
                        print(f"resume {rel(metric_path)}", flush=True)
                    else:
                        elapsed = run_command(command, log_path, args.dry_run)
                    if args.dry_run:
                        continue
                    if not metric_path.exists():
                        raise FileNotFoundError(metric_path)
                    record = {
                        "dataset": dataset,
                        "pred_len": args.pred_len,
                        "train_ratio": args.train_ratio,
                        "train_ratio_seed": args.train_ratio_seed,
                        "seed": seed,
                        "variant": variant,
                        "feature_path": rel(feature_path),
                        "source_metric_path": rel(source_metric),
                        "metric_path": rel(metric_path),
                        "seconds": elapsed,
                        **clean_metrics(metric_path),
                    }
                    records_by_key[(dataset, variant, seed)] = record
                    write_csv(run_csv, list(records_by_key.values()))
                except Exception as exc:
                    failures.append({
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "error": repr(exc),
                    })
                    write_csv(TABLES / f"{args.result_stem}_failures.csv", failures)
                    print(f"FAILED {dataset} {variant} seed={seed}: {exc}", flush=True)

    records = list(records_by_key.values())
    summary_rows = summarize(records)
    write_csv(run_csv, records)
    write_csv(TABLES / f"{args.result_stem}_summary.csv", summary_rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "pred_len": args.pred_len,
        "train_ratio": args.train_ratio,
        "train_ratio_seed": args.train_ratio_seed,
        "seeds": seeds,
        "records": records,
        "summary": summary_rows,
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{args.result_stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(TABLES / f"{args.result_stem}_summary.csv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
