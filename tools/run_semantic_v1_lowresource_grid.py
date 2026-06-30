from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

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

DATASET_ABBR = {
    "appliances_energy": "appl",
    "beijing_pm25": "pm25",
    "exchange_rate": "exch",
}


@dataclass(frozen=True)
class CaseConfig:
    dataset: str
    pred_len: int
    train_ratio: float
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    numeric_learning_rate: float
    train_epochs: int
    patience: int
    lambda_safety: float
    residual_budget_scale: float
    dropout: float


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in parse_csv_list(text)]


def parse_floats(text: str) -> list[float]:
    return [float(item) for item in parse_csv_list(text)]


def ratio_tag(ratio: float) -> str:
    return f"{ratio:.2f}".replace(".", "p")


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("-", "m").replace("+", "").replace(".", "p")


def abbr_dataset(dataset: str) -> str:
    return DATASET_ABBR.get(dataset, dataset.lower().replace("-", "").replace("_", "")[:10])


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def split_mode_for(dataset: str) -> str:
    return "ett_standard" if dataset in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"} else "ratio"


def setting_name(model_id: str, dataset: str, seq_len: int, pred_len: int, hidden_dim: int, sem_dim: int, des: str) -> str:
    return (
        f"long_term_forecast_{model_id}_CARE_Forecast_{dataset}_ftM"
        f"_sl{seq_len}_pl{pred_len}_hd{hidden_dim}_sd{sem_dim}_{des}_0"
    )


def legacy_epoch_tags(pred_len: int, ratio: float, stage1_default: int = 8, train_default: int = 8) -> tuple[int, int]:
    stage1 = stage1_default + (2 if pred_len >= 336 else 0)
    train = train_default + (2 if pred_len >= 336 else 0)
    if ratio <= 0.05:
        train += 1
    return stage1, train


def legacy_numeric_paths(dataset: str, seq_len: int, pred_len: int, ratio: float, hidden_dim: int, sem_dim: int, seed: int) -> tuple[Path, Path, str]:
    stage1_epochs, train_tag_epochs = legacy_epoch_tags(pred_len, ratio)
    tag = ratio_tag(ratio)
    short = abbr_dataset(dataset)
    prefix = f"{short}_p{pred_len}_r{tag}_e{stage1_epochs}x{train_tag_epochs}"
    model_id = f"{prefix}_NUM_S{seed}"
    des = f"{short}_p{pred_len}_r{tag}_num"
    setting = setting_name(model_id, dataset, seq_len, pred_len, hidden_dim, sem_dim, des)
    return ROOT / "checkpoints" / setting / "checkpoint.pth", ROOT / "outputs" / setting / "metrics.json", setting


def registry_numeric_paths(args: argparse.Namespace, dataset: str, pred_len: int, ratio: float) -> tuple[Path, Path, str] | None:
    registry_path = ROOT / args.numeric_registry
    if not registry_path.exists():
        return None
    df = pd.read_csv(registry_path)
    if df.empty:
        return None
    mask = (
        (df["dataset"].astype(str) == dataset)
        & (df["pred_len"].astype(int) == int(pred_len))
        & (df["train_ratio"].astype(float).round(6) == round(float(ratio), 6))
    )
    rows = df[mask].copy()
    if rows.empty:
        return None
    rows["mse"] = rows["mse"].astype(float)
    row = rows.sort_values("mse", ascending=True).iloc[0]
    ckpt = ROOT / str(row["checkpoint_path"])
    metric = ROOT / str(row["metric_path"])
    setting = str(row.get("setting", ckpt.parent.name))
    if not ckpt.exists() or not metric.exists():
        return None
    return ckpt, metric, setting


def semantic_artifact_paths(dataset: str, seq_len: int, pred_len: int) -> tuple[Path, Path]:
    prefix = f"{dataset}_sl{seq_len}_pl{pred_len}"
    return (
        ROOT / "generated" / dataset / f"{prefix}_text_M_semantic_v1.csv",
        ROOT / "generated" / dataset / f"{prefix}_semantic_v1_text_features.npz",
    )


def run_command(cmd: list[str], dry_run: bool = False) -> float:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0.0
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return seconds


def prepare_artifacts(args: argparse.Namespace, dataset: str, pred_len: int) -> tuple[Path, Path]:
    text_path, feature_path = semantic_artifact_paths(dataset, args.seq_len, pred_len)
    if text_path.exists() and feature_path.exists() and not args.force_artifacts:
        print(f"reuse semantic artifacts: {text_path} / {feature_path}", flush=True)
        return text_path, feature_path
    cmd = [
        args.python,
        "tools/prepare_semantic_template_text_artifacts.py",
        "--python",
        args.python,
        "--datasets",
        dataset,
        "--pred-lens",
        str(pred_len),
        "--seq-len",
        str(args.seq_len),
        "--target",
        args.target,
        "--text-dim",
        str(args.text_dim),
    ]
    if args.force_artifacts:
        cmd += ["--force-text", "--force-features"]
    if args.dry_run:
        cmd.append("--dry-run")
    run_command(cmd, dry_run=False)
    if not args.dry_run and (not text_path.exists() or not feature_path.exists()):
        raise FileNotFoundError(f"Missing semantic artifacts: {text_path} / {feature_path}")
    return text_path, feature_path


def choose_case_config(dataset: str, pred_len: int, ratio: float, args: argparse.Namespace) -> CaseConfig:
    lower = dataset.lower()
    high_memory = lower in {"weather", "wind", "appliances_energy"} or pred_len >= 336
    medium_memory = lower in {"aqshunyi", "aqwan", "beijing_pm25", "czelan", "zafnoo"}

    if args.force_batch_size > 0:
        batch_size = args.force_batch_size
    elif high_memory:
        batch_size = 32
    elif medium_memory:
        batch_size = 48 if pred_len <= 192 else 32
    else:
        batch_size = 64

    if args.force_eval_batch_size > 0:
        eval_batch_size = args.force_eval_batch_size
    elif batch_size <= 32:
        eval_batch_size = 64
    else:
        eval_batch_size = 128

    lr = args.learning_rate
    safety = 1.0
    budget = 1.0
    dropout = 0.10
    if lower in {"exchange_rate", "etth2"}:
        lr = min(lr, 3e-4)
        budget = 0.75
    elif lower in {"weather", "wind"}:
        lr = min(lr, 4e-4)
        safety = 1.0
    elif lower in {"appliances_energy", "flight", "czelan"}:
        lr = max(lr, 5e-4)
        safety = 0.8
    elif lower in {"aqshunyi", "aqwan", "beijing_pm25", "zafnoo"}:
        lr = 5e-4
        budget = 1.0

    if pred_len >= 720:
        lr *= 0.8
    if ratio <= 0.05:
        dropout = 0.12

    train_epochs = args.train_epochs + (2 if pred_len >= 336 else 0)
    if ratio <= 0.05:
        train_epochs += 1
    patience = args.patience + (1 if pred_len >= 336 else 0)

    return CaseConfig(
        dataset=dataset,
        pred_len=pred_len,
        train_ratio=ratio,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=lr,
        numeric_learning_rate=args.numeric_learning_rate,
        train_epochs=train_epochs,
        patience=patience,
        lambda_safety=safety,
        residual_budget_scale=budget,
        dropout=dropout,
    )


def summarize_metric(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clean = data["test_clean"]
    corrupt_keys = [
        "test_missing_text",
        "test_noisy_text",
        "test_contradictory_text",
        "test_time_shift_text",
        "test_irrelevant_text",
    ]
    corrupt = [data[key] for key in corrupt_keys if key in data]
    robust_mse = sum(float(row["student_mse"]) for row in corrupt) / max(1, len(corrupt))
    robust_delta = sum(float(row["student_mse"] - row["numeric_base_mse"]) for row in corrupt) / max(1, len(corrupt))
    return {
        "student_mse": float(clean["student_mse"]),
        "student_mae": float(clean["student_mae"]),
        "numeric_base_mse": float(clean["numeric_base_mse"]),
        "numeric_base_mae": float(clean["numeric_base_mae"]),
        "teacher_oracle_mse": float(clean.get("teacher_oracle_mse", clean["numeric_base_mse"])),
        "teacher_oracle_mae": float(clean.get("teacher_oracle_mae", clean["numeric_base_mae"])),
        "gain": float(clean["numeric_base_mse"] - clean["student_mse"]),
        "gain_pct": float((clean["numeric_base_mse"] - clean["student_mse"]) / max(clean["numeric_base_mse"], 1e-12) * 100.0),
        "ntr": float(clean.get("student_ntr", float("nan"))),
        "ptr": float(clean.get("student_ptr", clean.get("student_positive_transfer_rate", float("nan")))),
        "gate": float(clean.get("student_gate", clean.get("mean_gate", float("nan")))),
        "reliability": float(clean.get("student_reliability", clean.get("mean_reliability", float("nan")))),
        "robust_mse_avg": float(robust_mse),
        "robust_delta_vs_base_avg": float(robust_delta),
        "residual_calibration_scale": float(clean.get("residual_calibration_scale", float("nan"))),
        "residual_calibration_threshold": float(clean.get("residual_calibration_threshold", float("nan"))),
    }


def common_args(args: argparse.Namespace, case: CaseConfig, text_path: Path, feature_path: Path) -> list[str]:
    return [
        args.python,
        "run.py",
        "--is_training",
        "1",
        "--model",
        "CARE_Forecast",
        "--data",
        case.dataset,
        "--root_path",
        ".",
        "--data_path",
        f"dataset/{case.dataset}.csv",
        "--text_path",
        rel(text_path),
        "--text_feature_path",
        rel(feature_path),
        "--features",
        "M",
        "--target",
        args.target,
        "--split_mode",
        split_mode_for(case.dataset),
        "--seq_len",
        str(args.seq_len),
        "--label_len",
        "48",
        "--pred_len",
        str(case.pred_len),
        "--text_dim",
        str(args.text_dim),
        "--student_text_col",
        "llm_history_text",
        "--student_text_cols",
        "llm_history_text",
        "llm_history_prior_text",
        "--teacher_text_col",
        "llm_future_text",
        "--teacher_text_cols",
        "llm_future_text",
        "llm_residual_text",
        "--hidden_dim",
        str(args.hidden_dim),
        "--sem_dim",
        str(args.sem_dim),
        "--dropout",
        f"{case.dropout:g}",
        "--use_revin",
        "0",
        "--frft_patch_len",
        str(args.seq_len),
        "--decomp_type",
        "moving_avg",
        "--moving_avg_kernel",
        "7",
        "--spectral_mix_init",
        "-1.5",
        "--batch_size",
        str(case.batch_size),
        "--eval_batch_size",
        str(case.eval_batch_size),
        "--patience",
        str(case.patience),
        "--learning_rate",
        f"{case.learning_rate:g}",
        "--numeric_learning_rate",
        f"{case.numeric_learning_rate:g}",
        "--weight_decay",
        f"{args.weight_decay:g}",
        "--positive_text_cols",
        "llm_history_text",
        "llm_history_prior_text",
        "history_text",
        "compact_text",
        "--negative_text_cols",
        "contradictory_text",
        "noisy_text",
        "missing_text",
        "time_shift_text",
        "irrelevant_text",
        "--train_ratio",
        f"{case.train_ratio:.2f}",
        "--train_ratio_seed",
        str(args.seed),
        "--train_ratio_mode",
        "uniform",
        "--seed",
        str(args.seed),
        "--use_gpu",
        "True",
        "--gpu",
        str(args.gpu),
    ]


def ensure_numeric_checkpoint(args: argparse.Namespace, case: CaseConfig, text_path: Path, feature_path: Path) -> tuple[Path, Path, str, float]:
    registry = registry_numeric_paths(args, case.dataset, case.pred_len, case.train_ratio)
    if registry is not None:
        ckpt, metric, setting = registry
        print(f"reuse registry numeric checkpoint: {setting}", flush=True)
        return ckpt, metric, setting, 0.0

    ckpt, metric, setting = legacy_numeric_paths(
        case.dataset,
        args.seq_len,
        case.pred_len,
        case.train_ratio,
        args.hidden_dim,
        args.sem_dim,
        args.seed,
    )
    if ckpt.exists() and metric.exists():
        print(f"reuse numeric checkpoint: {setting}", flush=True)
        return ckpt, metric, setting, 0.0

    stage1_epochs, train_tag_epochs = legacy_epoch_tags(case.pred_len, case.train_ratio)
    tag = ratio_tag(case.train_ratio)
    short = abbr_dataset(case.dataset)
    prefix = f"{short}_p{case.pred_len}_r{tag}_e{stage1_epochs}x{train_tag_epochs}"
    model_id = f"{prefix}_NUM_S{args.seed}"
    des = f"{short}_p{case.pred_len}_r{tag}_num"
    cmd = common_args(args, case, text_path, feature_path) + [
        "--model_id",
        model_id,
        "--method_profile",
        "numeric_only",
        "--ablation",
        "numeric_only",
        "--loss_fn",
        args.numeric_loss_fn,
        "--val_metric",
        args.numeric_val_metric,
        "--train_epochs",
        str(stage1_epochs),
        "--des",
        des,
    ]
    seconds = run_command(cmd, dry_run=args.dry_run)
    if not args.dry_run and (not ckpt.exists() or not metric.exists()):
        raise FileNotFoundError(f"Missing numeric outputs after training: {ckpt} / {metric}")
    return ckpt, metric, setting, seconds


def run_case(args: argparse.Namespace, dataset: str, pred_len: int, ratio: float) -> dict[str, Any]:
    case = choose_case_config(dataset, pred_len, ratio, args)
    text_path, feature_path = prepare_artifacts(args, dataset, pred_len)
    numeric_ckpt, numeric_metric, numeric_setting_name, numeric_seconds = ensure_numeric_checkpoint(args, case, text_path, feature_path)

    tag = ratio_tag(ratio)
    short = abbr_dataset(dataset)
    model_id = (
        f"{short}_sv1_p{pred_len}_r{tag}_bs{case.batch_size}_lr{lr_tag(case.learning_rate)}"
        f"_S{args.seed}"
    )
    des = f"{short}_sv1_p{pred_len}_r{tag}_bs{case.batch_size}_lr{lr_tag(case.learning_rate)}"
    setting = setting_name(model_id, dataset, args.seq_len, pred_len, args.hidden_dim, args.sem_dim, des)
    metric_path = ROOT / args.output_dir / setting / "metrics.json"

    if args.resume and metric_path.exists():
        print(f"resume semantic-v1 full: {setting}", flush=True)
        full_seconds = 0.0
    elif args.collect_existing:
        print(f"skip missing semantic-v1 full: {setting}", flush=True)
        return {}
    else:
        cmd = common_args(args, case, text_path, feature_path) + [
            "--model_id",
            model_id,
            "--method_profile",
            args.method_profile,
            "--ablation",
            "full",
            "--loss_fn",
            args.full_loss_fn,
            "--val_metric",
            args.full_val_metric,
            "--train_epochs",
            str(case.train_epochs),
            "--pretrained_numeric_checkpoint",
            rel(numeric_ckpt),
            "--freeze_numeric_backbone",
            "1" if args.freeze_numeric_backbone else "0",
            "--calibrate_residual",
            "1",
            "--calibration_score",
            "rel_gate",
            "--lambda_safety",
            f"{case.lambda_safety:g}",
            "--residual_budget_scale",
            f"{case.residual_budget_scale:g}",
            "--use_base_context",
            "1",
            "--use_quality_aug",
            "1",
            "--lambda_quality",
            f"{args.lambda_quality:g}",
            "--use_text_modulation",
            str(int(args.use_text_modulation)),
            "--text_mod_mode",
            args.text_mod_mode,
            "--text_mod_scale",
            f"{args.text_mod_scale:g}",
            "--des",
            des,
            "--output_dir",
            args.output_dir,
        ]
        full_seconds = run_command(cmd, dry_run=args.dry_run)

    if args.dry_run:
        return {}
    if not metric_path.exists():
        raise FileNotFoundError(metric_path)
    full_metrics = summarize_metric(metric_path)
    numeric_metrics = summarize_metric(numeric_metric)
    return {
        "dataset": dataset,
        "features": "M",
        "seq_len": args.seq_len,
        "pred_len": pred_len,
        "train_ratio": ratio,
        "split_mode": split_mode_for(dataset),
        "text_protocol": "semantic_v1",
        "student_text_cols": "llm_history_text+llm_history_prior_text",
        "teacher_text_cols": "llm_future_text+llm_residual_text",
        "method_profile": args.method_profile,
        "freeze_numeric_backbone": int(args.freeze_numeric_backbone),
        "train_epochs": case.train_epochs,
        "batch_size": case.batch_size,
        "eval_batch_size": case.eval_batch_size,
        "learning_rate": case.learning_rate,
        "numeric_learning_rate": case.numeric_learning_rate,
        "patience": case.patience,
        "lambda_safety": case.lambda_safety,
        "residual_budget_scale": case.residual_budget_scale,
        "dropout": case.dropout,
        "numeric_seconds": numeric_seconds,
        "full_seconds": full_seconds,
        "numeric_setting": numeric_setting_name,
        "numeric_metric_path": rel(numeric_metric),
        "full_metric_path": rel(metric_path),
        "text_path": rel(text_path),
        "text_feature_path": rel(feature_path),
        "numeric_only_mse": numeric_metrics["student_mse"],
        "numeric_only_mae": numeric_metrics["student_mae"],
        "student_mse": full_metrics["student_mse"],
        "student_mae": full_metrics["student_mae"],
        "teacher_oracle_mse": full_metrics["teacher_oracle_mse"],
        "teacher_oracle_mae": full_metrics["teacher_oracle_mae"],
        "gain": full_metrics["gain"],
        "gain_pct": full_metrics["gain_pct"],
        "ntr": full_metrics["ntr"],
        "ptr": full_metrics["ptr"],
        "gate": full_metrics["gate"],
        "reliability": full_metrics["reliability"],
        "robust_mse_avg": full_metrics["robust_mse_avg"],
        "robust_delta_vs_base_avg": full_metrics["robust_delta_vs_base_avg"],
        "residual_calibration_scale": full_metrics["residual_calibration_scale"],
        "residual_calibration_threshold": full_metrics["residual_calibration_threshold"],
    }


def write_outputs(rows: list[dict[str, Any]], failures: list[dict[str, Any]], args: argparse.Namespace) -> None:
    tables = ROOT / "tables"
    outputs = ROOT / args.output_dir
    tables.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    stem = args.result_stem
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "datasets": parse_csv_list(args.datasets),
            "pred_lens": parse_ints(args.pred_lens),
            "ratios": parse_floats(args.ratios),
            "seq_len": args.seq_len,
            "features": "M",
            "target": args.target,
            "text_protocol": "semantic_v1",
            "method_profile": args.method_profile,
            "freeze_numeric_backbone": bool(args.freeze_numeric_backbone),
        },
        "records": rows,
        "failures": failures,
    }
    (outputs / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        csv_path = tables / f"{stem}.csv"
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        df = pd.DataFrame(rows)
        dataset_summary = (
            df.groupby("dataset", as_index=False)
            .agg(
                cases=("dataset", "size"),
                numeric_mse=("numeric_only_mse", "mean"),
                student_mse=("student_mse", "mean"),
                teacher_mse=("teacher_oracle_mse", "mean"),
                gain_pct=("gain_pct", "mean"),
                ntr=("ntr", "mean"),
                reliability=("reliability", "mean"),
            )
            .sort_values("gain_pct", ascending=False)
        )
        ratio_horizon_summary = (
            df.groupby(["train_ratio", "pred_len"], as_index=False)
            .agg(
                cases=("dataset", "size"),
                numeric_mse=("numeric_only_mse", "mean"),
                student_mse=("student_mse", "mean"),
                teacher_mse=("teacher_oracle_mse", "mean"),
                gain_pct=("gain_pct", "mean"),
                ntr=("ntr", "mean"),
            )
            .sort_values(["train_ratio", "pred_len"])
        )
        dataset_summary.to_csv(tables / f"{stem}_dataset_summary.csv", index=False, encoding="utf-8-sig")
        ratio_horizon_summary.to_csv(tables / f"{stem}_ratio_horizon_summary.csv", index=False, encoding="utf-8-sig")

        lines = [
            "# Semantic-v1 Low-Resource Results",
            "",
            f"- generated_at: {payload['generated_at']}",
            f"- completed_cases: {len(rows)} / {len(parse_csv_list(args.datasets)) * len(parse_ints(args.pred_lens)) * len(parse_floats(args.ratios))}",
            f"- text: deployable `llm_history_text + llm_history_prior_text`; teacher-only `llm_future_text + llm_residual_text`.",
            f"- training: method_profile={args.method_profile}; freeze_numeric_backbone={int(args.freeze_numeric_backbone)}; calibrated residual on validation.",
            "",
            "| dataset | pred_len | ratio | bs | lr | numeric MSE | student MSE | gain % | teacher MSE | NTR | gate | rel | cal scale | metric |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in sorted(rows, key=lambda item: (item["dataset"], item["pred_len"], item["train_ratio"])):
            lines.append(
                f"| {row['dataset']} | {row['pred_len']} | {row['train_ratio']:.0%} | {row['batch_size']} | "
                f"{row['learning_rate']:.1e} | {row['numeric_only_mse']:.6f} | {row['student_mse']:.6f} | "
                f"{row['gain_pct']:+.2f}% | {row['teacher_oracle_mse']:.6f} | {row['ntr']:.3f} | "
                f"{row['gate']:.3f} | {row['reliability']:.3f} | {row['residual_calibration_scale']:.2f} | "
                f"`{row['full_metric_path']}` |"
            )
        (tables / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        summary_lines = [
            "# Semantic-v1 Dataset Summary",
            "",
            "| dataset | cases | numeric MSE | student MSE | teacher MSE | gain % | NTR | rel |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in dataset_summary.to_dict("records"):
            summary_lines.append(
                f"| {row['dataset']} | {int(row['cases'])} | {row['numeric_mse']:.6f} | "
                f"{row['student_mse']:.6f} | {row['teacher_mse']:.6f} | {row['gain_pct']:+.2f}% | "
                f"{row['ntr']:.3f} | {row['reliability']:.3f} |"
            )
        (tables / f"{stem}_dataset_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if failures:
        with (tables / f"{stem}_failures.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run formal semantic-v1 CARE-Forecast low-resource grid.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--pred-lens", default="96,192,336,720")
    parser.add_argument("--ratios", default="0.05,0.10,0.20")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--target", default="OT")
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--sem-dim", type=int, default=128)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--numeric-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--numeric-loss-fn", default="mae")
    parser.add_argument("--numeric-val-metric", default="loss")
    parser.add_argument("--full-loss-fn", default="mae")
    parser.add_argument("--full-val-metric", default="loss")
    parser.add_argument("--method-profile", default="privileged_bridge")
    parser.add_argument("--numeric-registry", default="tables/numeric_base_registry.csv")
    parser.add_argument("--freeze-numeric-backbone", type=int, default=1)
    parser.add_argument("--lambda-quality", type=float, default=0.02)
    parser.add_argument("--use-text-modulation", type=int, default=0)
    parser.add_argument("--text-mod-mode", default="film")
    parser.add_argument("--text-mod-scale", type=float, default=0.2)
    parser.add_argument("--force-batch-size", type=int, default=0)
    parser.add_argument("--force-eval-batch-size", type=int, default=0)
    parser.add_argument("--output-dir", default="./outputs_semantic_v1")
    parser.add_argument("--result-stem", default="semantic_v1_lowresource_results")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--collect-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for dataset in parse_csv_list(args.datasets):
        for pred_len in parse_ints(args.pred_lens):
            for ratio in parse_floats(args.ratios):
                try:
                    row = run_case(args, dataset, pred_len, ratio)
                except Exception as exc:
                    failure = {
                        "dataset": dataset,
                        "pred_len": pred_len,
                        "train_ratio": ratio,
                        "error": repr(exc),
                        "failed_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    failures.append(failure)
                    write_outputs(rows, failures, args)
                    print(f"FAILED {dataset} pred_len={pred_len} ratio={ratio}: {exc}", flush=True)
                    if not args.continue_on_error:
                        raise
                    continue
                if row:
                    rows.append(row)
                    write_outputs(rows, failures, args)
    write_outputs(rows, failures, args)
    print(ROOT / "tables" / f"{args.result_stem}.csv")
    print(ROOT / "tables" / f"{args.result_stem}.md")
    print(ROOT / args.output_dir / f"{args.result_stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
