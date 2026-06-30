from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


UNRELATED_SNIPPETS = [
    "A local sports team announced a new training schedule unrelated to the monitored system.",
    "A museum opened a historical exhibition with no connection to transformer operation.",
    "A recipe article discussed seasonal vegetables and kitchen preparation tips.",
    "A travel blog described sightseeing routes and hotel recommendations.",
    "An entertainment report summarized a film festival and celebrity interviews.",
]


def trend_label(values: np.ndarray) -> tuple[str, float]:
    if len(values) < 2:
        return "stable", 0.0
    x = np.arange(len(values), dtype=np.float64)
    slope = float(np.polyfit(x, values.astype(np.float64), 1)[0])
    scale = float(np.std(values) + 1e-6)
    score = slope * len(values) / scale
    if score > 0.35:
        return "upward", score
    if score < -0.35:
        return "downward", score
    return "stable", score


def change_label(values: np.ndarray) -> tuple[str, float]:
    if len(values) < 4:
        return "little recent change", 0.0
    half = max(1, len(values) // 4)
    prev = float(np.mean(values[-2 * half : -half]))
    recent = float(np.mean(values[-half:]))
    scale = float(np.std(values) + 1e-6)
    score = (recent - prev) / scale
    if score > 0.35:
        return "recent increase", score
    if score < -0.35:
        return "recent decrease", score
    return "little recent change", score


def volatility_label(values: np.ndarray) -> tuple[str, float]:
    mean_abs = float(abs(np.mean(values)) + 1e-6)
    ratio = float(np.std(values) / mean_abs)
    if ratio > 0.25:
        return "high", ratio
    if ratio > 0.10:
        return "medium", ratio
    return "low", ratio


def anomaly_label(values: np.ndarray) -> tuple[str, int]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)) + 1e-6)
    robust_z = np.abs(values - median) / (1.4826 * mad)
    count = int(np.sum(robust_z > 3.5))
    return ("yes" if count > 0 else "no"), count


def periodicity_label(values: np.ndarray, lags: tuple[int, ...] = (24, 12, 6)) -> tuple[str, float, int]:
    best_corr = 0.0
    best_lag = 0
    centered = values.astype(np.float64) - float(np.mean(values))
    for lag in lags:
        if len(values) <= 2 * lag:
            continue
        a = centered[:-lag]
        b = centered[lag:]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        corr = float(np.dot(a, b) / denom)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    strength = abs(best_corr)
    if strength > 0.55:
        return "strong", best_corr, best_lag
    if strength > 0.30:
        return "medium", best_corr, best_lag
    return "weak", best_corr, best_lag


def uncertainty_label(trend_score: float, vol_score: float, anomaly_count: int) -> str:
    if anomaly_count > 0 or vol_score > 0.25 or abs(trend_score) < 0.15:
        return "high"
    if vol_score > 0.10 or abs(trend_score) < 0.45:
        return "medium"
    return "low"


def top_correlations(history: np.ndarray, columns: list[str], target_idx: int, k: int = 2) -> tuple[list[str], list[str]]:
    target = history[:, target_idx]
    target = target - float(np.mean(target))
    pairs: list[tuple[float, str]] = []
    for idx, name in enumerate(columns):
        if idx == target_idx:
            continue
        series = history[:, idx] - float(np.mean(history[:, idx]))
        denom = float(np.linalg.norm(target) * np.linalg.norm(series) + 1e-8)
        corr = float(np.dot(target, series) / denom)
        pairs.append((corr, name))
    pairs.sort(key=lambda item: item[0], reverse=True)
    positive = [name for corr, name in pairs[:k] if corr > 0.15]
    negative = [name for corr, name in pairs[-k:] if corr < -0.15]
    return positive, negative


def summarize_segment(values: np.ndarray) -> dict[str, object]:
    trend, trend_score = trend_label(values)
    change, change_score = change_label(values)
    vol, vol_score = volatility_label(values)
    anomaly, anomaly_count = anomaly_label(values)
    periodicity, period_corr, period_lag = periodicity_label(values)
    uncertainty = uncertainty_label(trend_score, vol_score, anomaly_count)
    return {
        "trend": trend,
        "trend_score": round(float(trend_score), 4),
        "recent_change": change,
        "recent_change_score": round(float(change_score), 4),
        "volatility": vol,
        "volatility_score": round(float(vol_score), 4),
        "anomaly": anomaly,
        "anomaly_count": anomaly_count,
        "periodicity": periodicity,
        "period_corr": round(float(period_corr), 4),
        "period_lag": period_lag,
        "uncertainty": uncertainty,
        "first": round(float(values[0]), 4),
        "last": round(float(values[-1]), 4),
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
    }


def describe_history(summary: dict[str, object], positives: list[str], negatives: list[str], target: str) -> str:
    pos_text = ", ".join(positives) if positives else "no strongly positive auxiliary variable"
    neg_text = ", ".join(negatives) if negatives else "no strongly negative auxiliary variable"
    return (
        f"The observed history of {target} shows a {summary['trend']} trend with "
        f"{summary['volatility']} volatility and {summary['periodicity']} periodicity. "
        f"The recent pattern indicates {summary['recent_change']}. "
        f"Anomaly detected: {summary['anomaly']}. "
        f"Variables moving positively with {target}: {pos_text}. "
        f"Variables moving negatively with {target}: {neg_text}. "
        f"Uncertainty is {summary['uncertainty']}."
    )


def describe_future(summary: dict[str, object], target: str) -> str:
    return (
        f"Teacher-only future pattern for {target}: the future segment has a "
        f"{summary['trend']} trend, {summary['volatility']} volatility, "
        f"{summary['periodicity']} periodicity, and {summary['recent_change']}. "
        f"Anomaly detected: {summary['anomaly']}. "
        f"Future uncertainty is {summary['uncertainty']}."
    )


def describe_residual(summary: dict[str, object], target: str) -> str:
    direction = summary["trend"]
    if summary["mean"] > 0.05:
        correction = "positive correction"
    elif summary["mean"] < -0.05:
        correction = "negative correction"
    else:
        correction = "small correction"
    return (
        f"Teacher-only residual pattern for {target}: compared with a persistence baseline, "
        f"the future requires a {correction}. The residual trend is {direction}, "
        f"with {summary['volatility']} volatility and {summary['recent_change']}. "
        f"Residual uncertainty is {summary['uncertainty']}."
    )


def describe_compact(summary: dict[str, object]) -> str:
    keys = ["trend", "volatility", "periodicity", "recent_change", "anomaly", "uncertainty"]
    return "; ".join(f"{key}={summary[key]}" for key in keys)


def make_contradictory_text(summary: dict[str, object], target: str) -> str:
    opposite = {"upward": "downward", "downward": "upward", "stable": "strongly changing"}[str(summary["trend"])]
    change = {
        "recent increase": "recent decrease",
        "recent decrease": "recent increase",
        "little recent change": "a sharp recent change",
    }[str(summary["recent_change"])]
    return (
        f"Contradictory context: the observed history of {target} should be interpreted as "
        f"a {opposite} trend with {change}, despite the numerical window suggesting otherwise."
    )


def make_noisy_text(clean_text: str, row_id: int) -> str:
    snippet = UNRELATED_SNIPPETS[row_id % len(UNRELATED_SNIPPETS)]
    return f"{clean_text} Irrelevant inserted sentence: {snippet}"


def make_paraphrase(summary: dict[str, object], positives: list[str], negatives: list[str], target: str) -> str:
    pos_text = ", ".join(positives) if positives else "no clear positive companion"
    neg_text = ", ".join(negatives) if negatives else "no clear negative companion"
    return (
        f"For {target}, the past window is best summarized as {summary['trend']} and "
        f"{summary['volatility']} in variation. The short-term history shows "
        f"{summary['recent_change']}; periodic evidence is {summary['periodicity']}. "
        f"Related variables: positive [{pos_text}], negative [{neg_text}]."
    )


def is_multivariate_target(target: str) -> bool:
    return target.lower() in {"all", "__all__", "multi", "m", "multivariate"}


def summarize_multivariate_segment(segment: np.ndarray, columns: list[str]) -> dict[str, object]:
    per_var = {name: summarize_segment(segment[:, idx]) for idx, name in enumerate(columns)}
    trend_counts = {label: sum(1 for item in per_var.values() if item["trend"] == label) for label in ["upward", "downward", "stable"]}
    vol_scores = [float(item["volatility_score"]) for item in per_var.values()]
    anomaly_count = int(sum(int(item["anomaly_count"]) for item in per_var.values()))
    avg_trend = float(np.mean([float(item["trend_score"]) for item in per_var.values()]))
    avg_vol = float(np.mean(vol_scores))
    if trend_counts["upward"] > trend_counts["downward"] and trend_counts["upward"] >= trend_counts["stable"]:
        trend = "upward"
    elif trend_counts["downward"] > trend_counts["upward"] and trend_counts["downward"] >= trend_counts["stable"]:
        trend = "downward"
    else:
        trend = "stable"
    vol = "high" if avg_vol > 0.25 else "medium" if avg_vol > 0.10 else "low"
    uncertainty = uncertainty_label(avg_trend, avg_vol, anomaly_count)
    return {
        "trend": trend,
        "trend_score": round(avg_trend, 4),
        "recent_change": "mixed multivariate change",
        "recent_change_score": 0.0,
        "volatility": vol,
        "volatility_score": round(avg_vol, 4),
        "anomaly": "yes" if anomaly_count > 0 else "no",
        "anomaly_count": anomaly_count,
        "periodicity": "mixed",
        "period_corr": 0.0,
        "period_lag": 0,
        "uncertainty": uncertainty,
        "variables": per_var,
    }


def short_multivariate_profile(summary: dict[str, object], columns: list[str], max_vars: int = 7) -> str:
    per_var = summary["variables"]
    assert isinstance(per_var, dict)
    parts = []
    for name in columns[:max_vars]:
        item = per_var[name]
        parts.append(f"{name}: {item['trend']} trend, {item['volatility']} volatility, {item['recent_change']}")
    return "; ".join(parts)


def describe_multivariate_history(summary: dict[str, object], columns: list[str]) -> str:
    return (
        "The observed multivariate history shows "
        f"an overall {summary['trend']} tendency with {summary['volatility']} average volatility. "
        f"Variable profiles: {short_multivariate_profile(summary, columns)}. "
        f"Anomaly detected: {summary['anomaly']}. Overall uncertainty is {summary['uncertainty']}."
    )


def describe_multivariate_future(summary: dict[str, object], columns: list[str]) -> str:
    return (
        "Teacher-only future multivariate pattern: "
        f"the future segment has an overall {summary['trend']} tendency and {summary['volatility']} volatility. "
        f"Variable profiles: {short_multivariate_profile(summary, columns)}. "
        f"Future uncertainty is {summary['uncertainty']}."
    )


def describe_multivariate_residual(summary: dict[str, object], columns: list[str]) -> str:
    per_var = summary["variables"]
    assert isinstance(per_var, dict)
    corrections = []
    for name in columns:
        item = per_var[name]
        mean = float(item["mean"])
        if mean > 0.05:
            correction = "positive correction"
        elif mean < -0.05:
            correction = "negative correction"
        else:
            correction = "small correction"
        corrections.append(f"{name}: {correction}, {item['trend']} residual")
    return (
        "Teacher-only multivariate residual pattern relative to a persistence baseline: "
        f"{'; '.join(corrections)}. "
        f"Overall residual uncertainty is {summary['uncertainty']}."
    )


def make_multivariate_contradictory_text(summary: dict[str, object]) -> str:
    opposite = {"upward": "downward", "downward": "upward", "stable": "strongly changing"}[str(summary["trend"])]
    return (
        "Contradictory context: the observed multivariate history should be interpreted as "
        f"an overall {opposite} pattern with severe instability, despite the numerical window suggesting otherwise."
    )


def make_multivariate_paraphrase(summary: dict[str, object], columns: list[str]) -> str:
    return (
        "Across the variables, the past window can be summarized as "
        f"{summary['trend']} with {summary['volatility']} variation. "
        f"The main variable-level evidence is: {short_multivariate_profile(summary, columns)}."
    )


def build_multivariate_rows(df: pd.DataFrame, seq_len: int, pred_len: int, max_samples: int | None) -> pd.DataFrame:
    numeric_cols = [col for col in df.columns if col != "date"]
    values = df[numeric_cols].astype(float).to_numpy()
    total = len(df) - seq_len - pred_len + 1
    if total <= 0:
        raise ValueError("Dataset is shorter than seq_len + pred_len.")
    if max_samples is not None:
        total = min(total, max_samples)

    rows: list[dict[str, object]] = []
    for i in range(total):
        hist = values[i : i + seq_len]
        future = values[i + seq_len : i + seq_len + pred_len]
        naive = np.repeat(hist[-1:, :], pred_len, axis=0)
        residual = future - naive

        hist_summary = summarize_multivariate_segment(hist, numeric_cols)
        future_summary = summarize_multivariate_segment(future, numeric_cols)
        residual_summary = summarize_multivariate_segment(residual, numeric_cols)
        history_text = describe_multivariate_history(hist_summary, numeric_cols)
        future_text = describe_multivariate_future(future_summary, numeric_cols)
        residual_text = describe_multivariate_residual(residual_summary, numeric_cols)
        compact_text = (
            f"mode=multivariate; trend={hist_summary['trend']}; volatility={hist_summary['volatility']}; "
            f"anomaly={hist_summary['anomaly']}; uncertainty={hist_summary['uncertainty']}"
        )

        rows.append(
            {
                "sample_id": i,
                "start_idx": i,
                "end_idx": i + seq_len - 1,
                "pred_start_idx": i + seq_len,
                "pred_end_idx": i + seq_len + pred_len - 1,
                "start_date": df.loc[i, "date"],
                "end_date": df.loc[i + seq_len - 1, "date"],
                "pred_start_date": df.loc[i + seq_len, "date"],
                "pred_end_date": df.loc[i + seq_len + pred_len - 1, "date"],
                "history_text": history_text,
                "future_text": future_text,
                "residual_text": residual_text,
                "compact_text": compact_text,
                "paraphrase_text": make_multivariate_paraphrase(hist_summary, numeric_cols),
                "contradictory_text": make_multivariate_contradictory_text(hist_summary),
                "noisy_text": make_noisy_text(history_text, i),
                "missing_text": "No textual information available.",
                "history_summary": json.dumps(hist_summary, ensure_ascii=False),
                "future_summary": json.dumps(future_summary, ensure_ascii=False),
                "residual_summary": json.dumps(residual_summary, ensure_ascii=False),
                "positive_variables": json.dumps([], ensure_ascii=False),
                "negative_variables": json.dumps([], ensure_ascii=False),
            }
        )

    n = len(rows)
    for i, row in enumerate(rows):
        row["time_shift_text"] = rows[(i + pred_len * 4) % n]["history_text"]
        row["irrelevant_text"] = UNRELATED_SNIPPETS[i % len(UNRELATED_SNIPPETS)]
    return pd.DataFrame(rows)


def build_rows(df: pd.DataFrame, seq_len: int, pred_len: int, target: str, max_samples: int | None) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError("Input CSV must contain a date column.")
    numeric_cols = [col for col in df.columns if col != "date"]
    if is_multivariate_target(target):
        return build_multivariate_rows(df, seq_len=seq_len, pred_len=pred_len, max_samples=max_samples)
    if target not in numeric_cols:
        raise ValueError(f"Target {target!r} not found in numeric columns.")

    values = df[numeric_cols].astype(float).to_numpy()
    target_idx = numeric_cols.index(target)
    total = len(df) - seq_len - pred_len + 1
    if total <= 0:
        raise ValueError("Dataset is shorter than seq_len + pred_len.")
    if max_samples is not None:
        total = min(total, max_samples)

    rows: list[dict[str, object]] = []
    for i in range(total):
        hist = values[i : i + seq_len]
        future = values[i + seq_len : i + seq_len + pred_len, target_idx]
        target_hist = hist[:, target_idx]
        naive = np.full_like(future, target_hist[-1])
        residual = future - naive

        hist_summary = summarize_segment(target_hist)
        future_summary = summarize_segment(future)
        residual_summary = summarize_segment(residual)
        positives, negatives = top_correlations(hist, numeric_cols, target_idx)

        history_text = describe_history(hist_summary, positives, negatives, target)
        future_text = describe_future(future_summary, target)
        residual_text = describe_residual(residual_summary, target)
        compact_text = describe_compact(hist_summary)
        rows.append(
            {
                "sample_id": i,
                "start_idx": i,
                "end_idx": i + seq_len - 1,
                "pred_start_idx": i + seq_len,
                "pred_end_idx": i + seq_len + pred_len - 1,
                "start_date": df.loc[i, "date"],
                "end_date": df.loc[i + seq_len - 1, "date"],
                "pred_start_date": df.loc[i + seq_len, "date"],
                "pred_end_date": df.loc[i + seq_len + pred_len - 1, "date"],
                "history_text": history_text,
                "future_text": future_text,
                "residual_text": residual_text,
                "compact_text": compact_text,
                "paraphrase_text": make_paraphrase(hist_summary, positives, negatives, target),
                "contradictory_text": make_contradictory_text(hist_summary, target),
                "noisy_text": make_noisy_text(history_text, i),
                "missing_text": "No textual information available.",
                "history_summary": json.dumps(hist_summary, ensure_ascii=False),
                "future_summary": json.dumps(future_summary, ensure_ascii=False),
                "residual_summary": json.dumps(residual_summary, ensure_ascii=False),
                "positive_variables": json.dumps(positives, ensure_ascii=False),
                "negative_variables": json.dumps(negatives, ensure_ascii=False),
            }
        )

    # Deterministic time-shifted and irrelevant columns require all clean rows.
    n = len(rows)
    for i, row in enumerate(rows):
        row["time_shift_text"] = rows[(i + pred_len * 4) % n]["history_text"]
        row["irrelevant_text"] = UNRELATED_SNIPPETS[i % len(UNRELATED_SNIPPETS)]
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CARE text fields for a univariate forecasting setup.")
    parser.add_argument("--input", default="dataset/ETTh1.csv")
    parser.add_argument("--output", default="generated/ETTh1_sl96_pl96_text.csv")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--target", default="OT")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    rows = build_rows(df, seq_len=args.seq_len, pred_len=args.pred_len, target=args.target, max_samples=args.max_samples)
    rows.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote {output_path.resolve()} with {len(rows)} samples")
    print("Text columns: history_text, future_text, residual_text, compact_text, paraphrase_text, "
          "contradictory_text, noisy_text, missing_text, time_shift_text, irrelevant_text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
