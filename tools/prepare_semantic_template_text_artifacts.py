from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INTERVAL_TEXT = {
    "ETTh1": "1 hour",
    "ETTh2": "1 hour",
    "ETTm1": "15 minutes",
    "ETTm2": "15 minutes",
    "electricity": "1 hour",
    "weather": "10 minutes",
    "exchange_rate": "1 day",
}

DEFAULT_DATASETS = [
    "ETTh1",
    "ETTh2",
    "appliances_energy",
    "AQShunyi",
    "AQWan",
    "beijing_pm25",
    "CzeLan",
    "exchange_rate",
    "Flight",
    "weather",
    "Wind",
    "ZafNoo",
]

FEATURE_COLUMNS = [
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


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in parse_csv_list(text)]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def parse_json(value: Any, default: Any) -> Any:
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt_value(value: float, digits: int) -> str:
    value = safe_float(value)
    return f"{value:.{digits}f}"


def trend_label(delta: float, scale: float) -> str:
    threshold = 0.10 * max(scale, 1e-6)
    if delta > threshold:
        return "upward"
    if delta < -threshold:
        return "downward"
    return "stable"


def strength_label(value: float, scale: float) -> str:
    ratio = abs(value) / max(scale, 1e-6)
    if ratio >= 1.0:
        return "strong"
    if ratio >= 0.35:
        return "moderate"
    return "small"


def recent_relation(recent_mean: float, earlier_mean: float, scale: float) -> str:
    gap = recent_mean - earlier_mean
    threshold = 0.20 * max(scale, 1e-6)
    if gap > threshold:
        return "above"
    if gap < -threshold:
        return "below"
    return "close to"


def confidence_from_uncertainty(uncertainty: Any) -> str:
    value = str(uncertainty).strip().lower()
    if value == "low":
        return "high"
    if value == "medium":
        return "medium"
    return "low"


def variable_text(value: Any, fallback: str) -> str:
    names = parse_json(value, [])
    if not isinstance(names, list):
        names = []
    cleaned = [str(item) for item in names if str(item).strip()]
    return ", ".join(cleaned) if cleaned else fallback


def lag_text(summary: dict[str, Any]) -> str:
    lag = int(safe_float(summary.get("period_lag", 0), 0.0))
    if lag <= 0:
        return "no reliable lag"
    corr = safe_float(summary.get("period_corr", 0.0), 0.0)
    return f"{lag} (corr {corr:+.2f})"


def interval_for(dataset: str, raw: pd.DataFrame) -> str:
    if dataset in INTERVAL_TEXT:
        return INTERVAL_TEXT[dataset]
    try:
        dates = pd.to_datetime(raw["date"].head(8), errors="coerce")
        deltas = dates.diff().dropna()
        if not deltas.empty:
            seconds = float(deltas.dt.total_seconds().median())
            if 3500 <= seconds <= 3700:
                return "1 hour"
            if 85000 <= seconds <= 87000:
                return "1 day"
            if 550 <= seconds <= 650:
                return "10 minutes"
            if 850 <= seconds <= 950:
                return "15 minutes"
    except Exception:
        pass
    return "one step"


def build_history_text(
    dataset: str,
    row: Any,
    target_hist: np.ndarray,
    summary: dict[str, Any],
    positives: str,
    negatives: str,
    interval: str,
    digits: int,
) -> tuple[str, str, str]:
    first = float(target_hist[0])
    last = float(target_hist[-1])
    net = last - first
    scale = float(np.std(target_hist) + 1e-6)
    recent_len = min(24, len(target_hist))
    recent = target_hist[-recent_len:]
    earlier = target_hist[:-recent_len] if len(target_hist) > recent_len else target_hist[:recent_len]
    relation = recent_relation(float(np.mean(recent)), float(np.mean(earlier)), scale)
    label = trend_label(net, scale)
    confidence = confidence_from_uncertainty(summary.get("uncertainty"))
    volatility = str(summary.get("volatility", "medium"))
    lag = lag_text(summary)

    text = (
        f"From {row.start_date} to {row.end_date} sampled every {interval}, "
        f"the target changed from {fmt_value(first, digits)} to {fmt_value(last, digits)} "
        f"with net trend {fmt_value(net, digits)} ({label}). "
        f"Recent level is {relation} earlier history, variability is {volatility}, "
        f"and recurrence is strongest at lag {lag}. "
        f"Related variables: {positives}; opposite variables: {negatives}. "
        f"History-only confidence: {confidence}."
    )
    prior = (
        f"History-only prior from {row.start_date} to {row.end_date}: "
        f"the observed target is {label}, recent level is {relation} earlier history, "
        f"variability is {volatility}, and the most useful recurrence cue is lag {lag}. "
        f"Positive context: {positives}; opposite context: {negatives}. "
        f"Deployable confidence: {confidence}."
    )
    enriched = dict(summary)
    enriched.update(
        {
            "first": round(first, digits),
            "last": round(last, digits),
            "net_trend": round(net, digits),
            "trend": label,
            "recent_relation": relation,
            "confidence": confidence,
            "related_variables": positives,
            "opposite_variables": negatives,
        }
    )
    return text, prior, json.dumps(enriched, ensure_ascii=False)


def build_future_text(
    row: Any,
    target_future: np.ndarray,
    history_last: float,
    history_mean: float,
    history_std: float,
    summary: dict[str, Any],
    positives: str,
    negatives: str,
    interval: str,
    digits: int,
) -> tuple[str, str]:
    future_first = float(target_future[0])
    future_last = float(target_future[-1])
    future_mean = float(np.mean(target_future))
    future_change = future_last - future_first
    movement = trend_label(future_change, history_std)
    strength = strength_label(future_mean - history_last, history_std)
    future_relation = recent_relation(future_mean, history_mean, history_std)
    volatility = str(summary.get("volatility", "medium"))
    confidence = confidence_from_uncertainty(summary.get("uncertainty"))
    lag = lag_text(summary)
    text = (
        f"Teacher-only future context from {row.pred_start_date} to {row.pred_end_date} "
        f"sampled every {interval}: the target shows a {movement} future movement "
        f"with {strength} magnitude. Future variability is {volatility}, "
        f"the future level is {future_relation} the observed history, "
        f"and recurrence remains strongest at lag {lag}. "
        f"Future-related variables: {positives}; opposite future variables: {negatives}. "
        f"Privileged future confidence: {confidence}."
    )
    enriched = dict(summary)
    enriched.update(
        {
            "future_first": round(future_first, digits),
            "future_last": round(future_last, digits),
            "future_change": round(future_change, digits),
            "future_movement": movement,
            "future_strength": strength,
            "future_relation_to_history": future_relation,
            "confidence": confidence,
        }
    )
    return text, json.dumps(enriched, ensure_ascii=False)


def build_residual_text(
    row: Any,
    residual: np.ndarray,
    history_std: float,
    summary: dict[str, Any],
    positives: str,
    negatives: str,
) -> tuple[str, str]:
    mean_residual = float(np.mean(residual))
    abs_residual = float(np.mean(np.abs(residual)))
    if mean_residual > 0.05 * max(history_std, 1e-6):
        direction = "positive"
    elif mean_residual < -0.05 * max(history_std, 1e-6):
        direction = "negative"
    else:
        direction = "near-zero"
    strength = strength_label(abs_residual, history_std)
    volatility = str(summary.get("volatility", "medium"))
    confidence = confidence_from_uncertainty(summary.get("uncertainty"))
    lag = lag_text(summary)
    half = max(1, len(residual) // 2)
    early = float(np.mean(np.abs(residual[:half])))
    late = float(np.mean(np.abs(residual[half:]))) if len(residual) > half else early
    if early > late * 1.20:
        timing = "early"
    elif late > early * 1.20:
        timing = "late"
    else:
        timing = "spread across the window"
    text = (
        "Teacher-only residual context relative to the history-only baseline: "
        f"the future target requires a {direction} correction with {strength} magnitude. "
        f"Residual variability is {volatility}, and the correction is mainly {timing} "
        f"within the prediction window. Residual recurrence is strongest at lag {lag}. "
        f"Variables associated with positive correction: {positives}; "
        f"variables associated with negative correction: {negatives}. "
        f"Residual confidence: {confidence}."
    )
    enriched = dict(summary)
    enriched.update(
        {
            "correction_direction": direction,
            "correction_strength": strength,
            "correction_timing": timing,
            "mean_residual": round(mean_residual, 4),
            "mean_abs_residual": round(abs_residual, 4),
            "confidence": confidence,
        }
    )
    return text, json.dumps(enriched, ensure_ascii=False)


def add_semantic_columns(dataset: str, pred_len: int, args: argparse.Namespace) -> Path:
    prefix = f"{dataset}_sl{args.seq_len}_pl{pred_len}"
    data_csv = ROOT / "dataset" / f"{dataset}.csv"
    base_text = ROOT / "generated" / dataset / f"{prefix}_text_M.csv"
    output_csv = ROOT / "generated" / dataset / f"{prefix}_text_M_semantic_v1.csv"
    if not data_csv.exists():
        raise FileNotFoundError(data_csv)
    if not base_text.exists():
        run_command(
            [
                args.python,
                "tools/prepare_care_text.py",
                "--input",
                f"dataset/{dataset}.csv",
                "--output",
                rel(base_text),
                "--seq-len",
                str(args.seq_len),
                "--pred-len",
                str(pred_len),
                "--target",
                args.target,
            ],
            dry_run=args.dry_run,
        )
    if output_csv.exists() and not args.force_text:
        print(f"reuse {output_csv}", flush=True)
        return output_csv

    raw = pd.read_csv(data_csv)
    text_df = pd.read_csv(base_text)
    numeric_cols = [col for col in raw.columns if col != "date"]
    if args.target not in numeric_cols:
        raise ValueError(f"target {args.target!r} not found in {numeric_cols}")
    target_idx = numeric_cols.index(args.target)
    values = raw[numeric_cols].astype(float).to_numpy(dtype=np.float64)
    interval = interval_for(dataset, raw)

    history_texts: list[str] = []
    history_priors: list[str] = []
    future_texts: list[str] = []
    residual_texts: list[str] = []
    privileged_texts: list[str] = []
    history_summaries: list[str] = []
    future_summaries: list[str] = []
    residual_summaries: list[str] = []

    for row in text_df.itertuples(index=False):
        start = int(row.start_idx)
        hist = values[start : start + args.seq_len, target_idx]
        future = values[start + args.seq_len : start + args.seq_len + pred_len, target_idx]
        residual = future - hist[-1]
        hist_summary = parse_json(getattr(row, "history_summary", "{}"), {})
        future_summary = parse_json(getattr(row, "future_summary", "{}"), {})
        residual_summary = parse_json(getattr(row, "residual_summary", "{}"), {})
        positives = variable_text(getattr(row, "positive_variables", "[]"), "no strongly positive auxiliary variable")
        negatives = variable_text(getattr(row, "negative_variables", "[]"), "no strongly negative auxiliary variable")
        history_text, history_prior, history_summary = build_history_text(
            dataset,
            row,
            hist,
            hist_summary,
            positives,
            negatives,
            interval,
            args.digits,
        )
        future_text, future_summary = build_future_text(
            row,
            future,
            history_last=float(hist[-1]),
            history_mean=float(np.mean(hist)),
            history_std=float(np.std(hist) + 1e-6),
            summary=future_summary,
            positives=positives,
            negatives=negatives,
            interval=interval,
            digits=args.digits,
        )
        residual_text, residual_summary = build_residual_text(
            row,
            residual,
            history_std=float(np.std(hist) + 1e-6),
            summary=residual_summary,
            positives=positives,
            negatives=negatives,
        )

        history_texts.append(history_text)
        history_priors.append(history_prior)
        future_texts.append(future_text)
        residual_texts.append(residual_text)
        privileged_texts.append(f"{future_text} {residual_text}")
        history_summaries.append(history_summary)
        future_summaries.append(future_summary)
        residual_summaries.append(residual_summary)

    text_df["llm_history_text"] = history_texts
    text_df["llm_history_prior_text"] = history_priors
    text_df["llm_future_text"] = future_texts
    text_df["llm_residual_text"] = residual_texts
    text_df["llm_privileged_text"] = privileged_texts
    text_df["history_prior_summary"] = history_summaries
    text_df["history_numeric_summary"] = history_summaries
    text_df["history_future_summary"] = history_summaries
    text_df["future_summary"] = future_summaries
    text_df["residual_summary"] = residual_summaries
    text_df["semantic_text_protocol"] = (
        "semantic_v1: deployable history-only template plus teacher-only future/residual privileged templates; "
        "history text contains no future target values"
    )
    text_df["semantic_text_generator"] = "llm_designed_template_v1"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    text_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"wrote {output_csv} rows={len(text_df)}", flush=True)
    print(text_df["llm_history_text"].iloc[0], flush=True)
    return output_csv


def build_features(dataset: str, pred_len: int, text_csv: Path, args: argparse.Namespace) -> Path:
    prefix = f"{dataset}_sl{args.seq_len}_pl{pred_len}"
    output_npz = ROOT / "generated" / dataset / f"{prefix}_semantic_v1_text_features.npz"
    if output_npz.exists() and not args.force_features:
        print(f"reuse {output_npz}", flush=True)
        return output_npz
    run_command(
        [
            args.python,
            "tools/combine_text_features.py",
            "--text_csv",
            rel(text_csv),
            "--output",
            rel(output_npz),
            "--include_hybrid",
            "--hybrid_dim",
            str(args.text_dim),
            "--columns",
            *FEATURE_COLUMNS,
        ],
        dry_run=args.dry_run,
    )
    return output_npz


def prepare_one(dataset: str, pred_len: int, args: argparse.Namespace) -> tuple[Path, Path]:
    text_csv = add_semantic_columns(dataset, pred_len, args)
    feature_npz = build_features(dataset, pred_len, text_csv, args)
    return text_csv, feature_npz


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare semantic-v1 text artifacts for CARE-Forecast.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--pred-lens", default="96,192,336,720")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--target", default="OT")
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--force-text", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for dataset in parse_csv_list(args.datasets):
        for pred_len in parse_ints(args.pred_lens):
            text_csv, feature_npz = prepare_one(dataset, pred_len, args)
            print(f"prepared {dataset} pred_len={pred_len}: {text_csv} / {feature_npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
