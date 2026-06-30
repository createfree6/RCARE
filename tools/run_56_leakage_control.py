from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_55_ablation_grid as ab55


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "tables"
RATIO = ab55.RATIO
SEQ_LEN = ab55.SEQ_LEN

PRIV_COLS = ["llm_future_text", "llm_residual_text"]
STUDENT_COLS = ["llm_history_text", "llm_history_prior_text"]
LEAKAGE_PATTERNS = [
    "teacher-only",
    "future segment",
    "future pattern",
    "residual pattern",
    "residual summary",
    "compared with",
    "forecast horizon",
    "privileged",
]


def _metric_path(path_value: str) -> Path:
    path = ROOT / path_value
    return path if path.exists() else Path(path_value)


def _deploy_equiv(metric_path: Path) -> dict[str, float]:
    data = json.loads(metric_path.read_text(encoding="utf-8"))
    check = data.get("deploy_equivalence_check", {})
    return {k: float(check.get(k, 0.0)) for k in [
        "student_max_abs_diff",
        "base_max_abs_diff",
        "gate_max_abs_diff",
        "reliability_max_abs_diff",
    ]}


def _metrics_clean(path: Path) -> dict[str, float]:
    return ab55.metrics_clean(path)


def _row(
    dataset: str,
    pred_len: int,
    variant: str,
    label: str,
    metric_path: Path,
    base_mse: float,
    base_mae: float,
    source: str,
    seconds: float = 0.0,
) -> dict[str, Any]:
    m = _metrics_clean(metric_path)
    metric_base_mse = m["numeric_base_mse"]
    metric_base_mae = m["numeric_base_mae"]
    if source == "trained_control":
        base_mse = metric_base_mse
        base_mae = metric_base_mae
    deploy = _deploy_equiv(metric_path)
    return {
        "dataset": dataset,
        "pred_len": pred_len,
        "train_ratio": RATIO,
        "variant": variant,
        "variant_label": label,
        "student_mse": m["student_mse"],
        "student_mae": m["student_mae"],
        "numeric_base_mse": base_mse,
        "numeric_base_mae": base_mae,
        "gain_pct": (base_mse - m["student_mse"]) / max(base_mse, 1e-12) * 100.0,
        "mae_gain_pct": (base_mae - m["student_mae"]) / max(base_mae, 1e-12) * 100.0,
        "teacher_oracle_mse": m["teacher_oracle_mse"],
        "mean_gate": m["mean_gate"],
        "mean_reliability": m["mean_reliability"],
        **deploy,
        "metric_path": ab55.rel(metric_path),
        "source": source,
        "seconds": seconds,
    }


def _audit_text_leakage(dataset: str, pred_len: int) -> dict[str, Any]:
    text_path, _feature_path = ab55.text_artifacts(dataset, pred_len)
    df = pd.read_csv(text_path)
    rows: dict[str, Any] = {"dataset": dataset, "pred_len": pred_len, "rows": len(df)}
    for col in STUDENT_COLS:
        series = df[col].fillna("").astype(str) if col in df.columns else pd.Series([], dtype=str)
        lowered = series.str.lower()
        hits = lowered.map(lambda x: any(pattern in x for pattern in LEAKAGE_PATTERNS))
        rows[f"{col}_leakage_hits"] = int(hits.sum())
        rows[f"{col}_leakage_rate"] = float(hits.mean()) if len(hits) else math.nan
    for col in PRIV_COLS:
        series = df[col].fillna("").astype(str) if col in df.columns else pd.Series([], dtype=str)
        rows[f"{col}_nonempty"] = int(series.str.strip().ne("").sum())
    rows["audit_pass"] = all(rows.get(f"{col}_leakage_hits", 1) == 0 for col in STUDENT_COLS)
    return rows


def _make_shuffled_npz(dataset: str, pred_len: int, seed: int, output_dir: Path) -> Path:
    _text_path, feature_path = ab55.text_artifacts(dataset, pred_len)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}_sl{SEQ_LEN}_pl{pred_len}_semantic_v1_text_features_privileged_shuffled_s{seed}.npz"
    if out_path.exists():
        return out_path
    rng = np.random.default_rng(seed)
    with np.load(feature_path) as src:
        arrays: dict[str, np.ndarray] = {}
        n = None
        for key in src.files:
            value = src[key].astype(np.float32)
            if n is None:
                n = value.shape[0]
            arrays[key] = value.copy()
        if n is None:
            raise ValueError(f"No arrays in {feature_path}")
        perm = rng.permutation(n)
        if np.all(perm == np.arange(n)) and n > 1:
            perm = np.roll(perm, 1)
        for col in PRIV_COLS:
            if col not in arrays:
                raise ValueError(f"Missing privileged feature column {col} in {feature_path}")
            arrays[col] = arrays[col][perm]
        arrays["shuffle_permutation"] = perm.astype(np.int64)
    np.savez_compressed(out_path, **arrays)
    return out_path


def _control_command(
    args: argparse.Namespace,
    case: dict[str, Any],
    feature_path: Path,
    numeric_ckpt: Path,
    control_tag: str,
) -> tuple[list[str], Path, str]:
    dataset = str(case["dataset"])
    pred_len = int(case["pred_len"])
    text_path, _feature_path = ab55.text_artifacts(dataset, pred_len)
    short = ab55.abbr_dataset(dataset)
    tag = ab55.ratio_tag(RATIO)
    model_id = f"{short}_lc56_{control_tag}_p{pred_len}_r{tag}_s{args.seed}"
    des = f"lc56_{control_tag}_p{pred_len}"
    setting = ab55.setting_name(model_id, dataset, pred_len, args.hidden_dim, des)
    metric_path = ROOT / args.output_dir / setting / "metrics.json"
    base_cmd, _old_metric_path, _ = ab55.build_variant_command(
        args,
        case,
        "wo_distillation",
        text_path,
        feature_path,
        numeric_ckpt,
    )
    cmd = []
    skip_next = False
    replacements = {
        "--model_id": model_id,
        "--des": des,
        "--output_dir": args.output_dir,
        "--text_feature_path": ab55.rel(feature_path),
        "--lambda_distill": "0.5",
    }
    for i, item in enumerate(base_cmd):
        if skip_next:
            skip_next = False
            continue
        if item in replacements:
            cmd.extend([item, replacements[item]])
            skip_next = True
        else:
            cmd.append(item)
    return cmd, metric_path, setting


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Section 5.6 leakage and privileged-semantic control.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", default="AQShunyi,weather,Wind")
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--output-dir", default="./outputs_leakage_56")
    parser.add_argument("--result-stem", default="leakage_56_h96_results")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--numeric-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--default-calibration-scales", default="0,0.25,0.5,0.75,1,1.25,1.5")
    parser.add_argument("--default-calibration-thresholds", default="0,0.1,0.2,0.4,0.6,0.8,0.9")
    args = parser.parse_args()

    datasets = ab55.parse_csv(args.datasets)
    pred_len = int(args.pred_len)
    log_dir = ROOT / args.output_dir / "logs"
    shuffle_dir = ROOT / args.output_dir / "shuffled_features"
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dataset in datasets:
        cases = ab55.load_param_cases(dataset, {pred_len})
        if not cases:
            raise ValueError(f"No parameter-bank case for {dataset} pred_len={pred_len}")
        case = cases[0]
        audits.append(_audit_text_leakage(dataset, pred_len))
        text_path, feature_path = ab55.text_artifacts(dataset, pred_len)
        base_mse = float(case["numeric_mse"])
        numeric_metric = ab55.find_metric_for_setting(str(case["numeric_setting"]))
        numeric_ckpt, numeric_metric = ab55.ensure_numeric_checkpoint(args, case, text_path, feature_path, log_dir)
        base_mae = _metrics_clean(numeric_metric)["student_mae"]

        full_metric = _metric_path(str(case["full_metric_path"]))
        rows.append(_row(dataset, pred_len, "numeric_prior", "Numeric prior only", numeric_metric, base_mse, base_mae, "existing_numeric_metric"))
        rows.append(_row(dataset, pred_len, "full_rcare", "Full RCARE-Forecast", full_metric, base_mse, base_mae, f"positive_safe_significant:{case.get('action','')}"))

        try:
            cmd, metric_path, _setting = _control_command(args, case, feature_path, numeric_ckpt, "aligned_priv")
            if args.resume and metric_path.exists():
                seconds = 0.0
                print(f"resume {dataset} p{pred_len} aligned_privileged: {metric_path}", flush=True)
            else:
                seconds = ab55.run_command(cmd, log_dir / f"{dataset}_p{pred_len}_aligned_privileged.log", dry_run=args.dry_run)
            if not args.dry_run:
                if not metric_path.exists():
                    raise FileNotFoundError(metric_path)
                rows.append(_row(dataset, pred_len, "aligned_privileged", "Aligned privileged future/residual text", metric_path, base_mse, base_mae, "trained_control", seconds))
        except Exception as exc:
            failures.append({
                "dataset": dataset,
                "pred_len": pred_len,
                "variant": "aligned_privileged",
                "error": repr(exc),
                "failed_at": datetime.now().isoformat(timespec="seconds"),
            })
            print(f"FAILED {dataset} p{pred_len} aligned_privileged: {exc}", flush=True)

        try:
            shuffled_feature_path = _make_shuffled_npz(dataset, pred_len, args.seed, shuffle_dir)
            cmd, metric_path, _setting = _control_command(args, case, shuffled_feature_path, numeric_ckpt, "shuffle_priv")
            if args.resume and metric_path.exists():
                seconds = 0.0
                print(f"resume {dataset} p{pred_len} shuffled_privileged: {metric_path}", flush=True)
            else:
                seconds = ab55.run_command(cmd, log_dir / f"{dataset}_p{pred_len}_shuffled_privileged.log", dry_run=args.dry_run)
            if not args.dry_run:
                if not metric_path.exists():
                    raise FileNotFoundError(metric_path)
                rows.append(_row(dataset, pred_len, "shuffled_privileged", "Privileged future/residual text shuffled", metric_path, base_mse, base_mae, "trained_control", seconds))
        except Exception as exc:
            failures.append({
                "dataset": dataset,
                "pred_len": pred_len,
                "variant": "shuffled_privileged",
                "error": repr(exc),
                "failed_at": datetime.now().isoformat(timespec="seconds"),
            })
            print(f"FAILED {dataset} p{pred_len} shuffled_privileged: {exc}", flush=True)

        _write_csv(TABLE_DIR / f"{args.result_stem}.csv", rows)
        _write_csv(TABLE_DIR / f"{args.result_stem}_text_audit.csv", audits)
        if failures:
            _write_csv(TABLE_DIR / f"{args.result_stem}_failures.csv", failures)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "pred_len": pred_len,
        "ratio": RATIO,
        "records": rows,
        "text_audit": audits,
        "failures": failures,
    }
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.result_stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(TABLE_DIR / f"{args.result_stem}.csv", rows)
    _write_csv(TABLE_DIR / f"{args.result_stem}_text_audit.csv", audits)
    if failures:
        _write_csv(TABLE_DIR / f"{args.result_stem}_failures.csv", failures)
    print(TABLE_DIR / f"{args.result_stem}.csv")
    print(TABLE_DIR / f"{args.result_stem}_text_audit.csv")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
