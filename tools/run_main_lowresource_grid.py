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


DEFAULT_SKIP = {
    "electricity",
    "traffic",
    "METR-LA",
    "ETTm1",
    "ETTm2",
    "solar_AL",
}

DATASET_ABBR = {
    "appliances_energy": "appl",
    "beijing_pm25": "pm25",
    "exchange_rate": "exch",
}


@dataclass(frozen=True)
class RunCase:
    dataset: str
    pred_len: int
    train_ratio: float
    stage1_epochs: int
    train_epochs: int
    batch_size: int
    eval_batch_size: int
    split_mode: str


def ratio_tag(ratio: float) -> str:
    return f"{ratio:.2f}".replace(".", "p")


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in parse_list(text)]


def parse_floats(text: str) -> list[float]:
    return [float(item) for item in parse_list(text)]


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def setting_name(model_id: str, dataset: str, features: str, seq_len: int, pred_len: int, des: str) -> str:
    return (
        f"long_term_forecast_{model_id}_CARE_Forecast_{dataset}_ft{features}"
        f"_sl{seq_len}_pl{pred_len}_hd256_sd128_{des}_0"
    )


def abbr_dataset(dataset: str) -> str:
    return DATASET_ABBR.get(dataset, dataset.lower().replace("-", "").replace("_", "")[:10])


def run_command(cmd: list[str], cwd: Path, dry_run: bool = False) -> float:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0.0
    start = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return elapsed


def dataset_names(root: Path, skip: set[str]) -> list[str]:
    names = []
    for path in sorted((root / "dataset").glob("*.csv")):
        stem = path.stem
        if stem in skip:
            continue
        names.append(stem)
    return names


def split_mode_for(dataset: str) -> str:
    return "ett_standard" if dataset in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"} else "ratio"


def choose_batch_size(dataset: str, pred_len: int, root: Path) -> tuple[int, int]:
    csv_path = root / "dataset" / f"{dataset}.csv"
    cols = len([col for col in pd.read_csv(csv_path, nrows=1).columns if col != "date"])
    if dataset.lower() == "wind":
        return 32, 64
    if dataset.lower() == "weather":
        return 32, 64
    if pred_len >= 720:
        return 32, 64
    if pred_len >= 336 or cols >= 20:
        return 64, 128
    return 128, 256


def choose_epochs(pred_len: int, ratio: float, stage1_default: int, train_default: int) -> tuple[int, int]:
    # Keep the first formal grid tractable while giving long horizons enough optimization.
    stage1 = stage1_default + (2 if pred_len >= 336 else 0)
    train = train_default + (2 if pred_len >= 336 else 0)
    if ratio <= 0.05:
        train += 1
    return stage1, train


def prepare_artifacts(
    root: Path,
    python: str,
    dataset: str,
    seq_len: int,
    pred_len: int,
    text_dim: int,
    force: bool,
    dry_run: bool,
) -> tuple[Path, Path]:
    out_dir = root / "generated" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset}_sl{seq_len}_pl{pred_len}"
    text_path = out_dir / f"{prefix}_text_M.csv"
    feature_path = out_dir / f"{prefix}_hybrid_text_features.npz"

    if force or not text_path.exists():
        run_command(
            [
                python,
                "tools/prepare_care_text.py",
                "--input",
                f"dataset/{dataset}.csv",
                "--output",
                rel(text_path, root),
                "--seq-len",
                str(seq_len),
                "--pred-len",
                str(pred_len),
                "--target",
                "OT",
            ],
            root,
            dry_run=dry_run,
        )

    if force or not feature_path.exists():
        run_command(
            [
                python,
                "tools/combine_text_features.py",
                "--text_csv",
                rel(text_path, root),
                "--output",
                rel(feature_path, root),
                "--include_hybrid",
                "--hybrid_dim",
                str(text_dim),
            ],
            root,
            dry_run=dry_run,
        )
    return text_path, feature_path


def summarize_metric(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clean = data["test_clean"]
    corrupt_keys = [
        "test_contradictory_text",
        "test_noisy_text",
        "test_missing_text",
        "test_time_shift_text",
        "test_irrelevant_text",
    ]
    corrupt = [data[k] for k in corrupt_keys if k in data]
    robust_mse = float(sum(row["student_mse"] for row in corrupt) / len(corrupt)) if corrupt else float("nan")
    robust_delta = (
        float(sum(row["student_mse"] - row["numeric_base_mse"] for row in corrupt) / len(corrupt))
        if corrupt
        else float("nan")
    )
    return {
        "student_mse": float(clean["student_mse"]),
        "student_mae": float(clean["student_mae"]),
        "numeric_base_mse": float(clean["numeric_base_mse"]),
        "numeric_base_mae": float(clean["numeric_base_mae"]),
        "teacher_oracle_mse": float(clean["teacher_oracle_mse"]),
        "teacher_oracle_mae": float(clean["teacher_oracle_mae"]),
        "gain": float(clean["numeric_base_mse"] - clean["student_mse"]),
        "gain_pct": float((clean["numeric_base_mse"] - clean["student_mse"]) / max(clean["numeric_base_mse"], 1e-12) * 100.0),
        "ptr": float(clean.get("student_ptr", clean.get("student_positive_transfer_rate", float("nan")))),
        "ntr": float(clean.get("student_ntr", float("nan"))),
        "gate": float(clean.get("student_gate", clean.get("mean_gate", float("nan")))),
        "reliability": float(clean.get("student_reliability", clean.get("mean_reliability", float("nan")))),
        "robust_mse_avg": robust_mse,
        "robust_delta_vs_base_avg": robust_delta,
    }


def make_common_args(
    python: str,
    case: RunCase,
    root: Path,
    text_path: Path,
    feature_path: Path,
    text_dim: int,
    seq_len: int,
    seed: int,
    lr: float,
    patience: int,
    numeric_lr: float | None = None,
) -> list[str]:
    args = [
        python,
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
        rel(text_path, root),
        "--text_feature_path",
        rel(feature_path, root),
        "--features",
        "M",
        "--target",
        "OT",
        "--split_mode",
        case.split_mode,
        "--seq_len",
        str(seq_len),
        "--label_len",
        "48",
        "--pred_len",
        str(case.pred_len),
        "--text_dim",
        str(text_dim),
        "--student_text_col",
        "history_text",
        "--teacher_text_col",
        "future_text",
        "--teacher_text_cols",
        "future_text",
        "residual_text",
        "--hidden_dim",
        "256",
        "--sem_dim",
        "128",
        "--dropout",
        "0.1",
        "--use_revin",
        "0",
        "--frft_patch_len",
        str(seq_len),
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
        str(patience),
        "--learning_rate",
        f"{lr:g}",
        "--weight_decay",
        "1e-4",
        "--lambda_base",
        "0.2",
        "--lambda_phys",
        "0.001",
        "--positive_text_cols",
        "history_text",
        "paraphrase_text",
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
        str(seed),
        "--train_ratio_mode",
        "uniform",
        "--seed",
        str(seed),
        "--use_gpu",
        "True",
        "--gpu",
        "0",
    ]
    if numeric_lr is not None:
        args += ["--numeric_learning_rate", f"{numeric_lr:g}"]
    return args


def run_case(
    root: Path,
    python: str,
    case: RunCase,
    text_path: Path,
    feature_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tag = ratio_tag(case.train_ratio)
    dataset_short = abbr_dataset(case.dataset)
    prefix = f"{dataset_short}_p{case.pred_len}_r{tag}_e{case.stage1_epochs}x{case.train_epochs}"

    common = make_common_args(
        python,
        case,
        root,
        text_path,
        feature_path,
        args.text_dim,
        args.seq_len,
        args.seed,
        args.learning_rate,
        args.patience,
    )

    num_model_id = f"{prefix}_NUM_S{args.seed}"
    num_des = f"{dataset_short}_p{case.pred_len}_r{tag}_num"
    num_setting = setting_name(num_model_id, case.dataset, "M", args.seq_len, case.pred_len, num_des)
    num_ckpt = root / "checkpoints" / num_setting / "checkpoint.pth"
    num_metric = root / "outputs" / num_setting / "metrics.json"
    if args.resume and num_ckpt.exists() and num_metric.exists():
        print(f"Resume: numeric {num_setting}", flush=True)
        stage1_seconds = 0.0
    else:
        num_cmd = common + [
            "--model_id",
            num_model_id,
            "--method_profile",
            "numeric_only",
            "--ablation",
            "numeric_only",
            "--loss_fn",
            args.numeric_loss_fn,
            "--val_metric",
            args.numeric_val_metric,
            "--train_epochs",
            str(case.stage1_epochs),
            "--des",
            num_des,
        ]
        stage1_seconds = run_command(num_cmd, root, dry_run=args.dry_run)
    if args.dry_run:
        return {}
    if not num_ckpt.exists() or not num_metric.exists():
        raise FileNotFoundError(f"Missing numeric outputs for {num_setting}")
    num_metrics = summarize_metric(num_metric)

    full_common = make_common_args(
        python,
        case,
        root,
        text_path,
        feature_path,
        args.text_dim,
        args.seq_len,
        args.seed,
        args.learning_rate,
        args.patience,
        numeric_lr=args.numeric_learning_rate,
    )
    full_model_id = f"{prefix}_RCARE_WARM_S{args.seed}"
    full_des = f"{dataset_short}_p{case.pred_len}_r{tag}_warm"
    full_setting = setting_name(full_model_id, case.dataset, "M", args.seq_len, case.pred_len, full_des)
    full_metric = root / "outputs" / full_setting / "metrics.json"
    if args.resume and full_metric.exists():
        print(f"Resume: full {full_setting}", flush=True)
        train_seconds = 0.0
    else:
        full_cmd = full_common + [
            "--model_id",
            full_model_id,
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
            rel(num_ckpt, root),
            "--freeze_numeric_backbone",
            "0",
            "--des",
            full_des,
        ]
        train_seconds = run_command(full_cmd, root, dry_run=args.dry_run)
    full_metrics = summarize_metric(full_metric)
    return {
        "dataset": case.dataset,
        "features": "M",
        "seq_len": args.seq_len,
        "pred_len": case.pred_len,
        "train_ratio": case.train_ratio,
        "split_mode": case.split_mode,
        "text_path": rel(text_path, root),
        "text_feature_path": rel(feature_path, root),
        "stage1_epochs": case.stage1_epochs,
        "train_epochs": case.train_epochs,
        "batch_size": case.batch_size,
        "eval_batch_size": case.eval_batch_size,
        "stage1_seconds": stage1_seconds,
        "train_seconds": train_seconds,
        "numeric_metric_path": rel(num_metric, root),
        "full_metric_path": rel(full_metric, root),
        "numeric_only_mse": num_metrics["student_mse"],
        "numeric_only_mae": num_metrics["student_mae"],
        **{f"full_{k}": v for k, v in full_metrics.items()},
    }


def write_dataset_scripts(root: Path, datasets: list[str], args: argparse.Namespace) -> None:
    for dataset in datasets:
        script_dir = root / "scripts" / "datasets" / dataset
        script_dir.mkdir(parents=True, exist_ok=True)
        ps1 = script_dir / f"run_{dataset.lower()}_main_lowresource.ps1"
        sh = script_dir / f"run_{dataset.lower()}_main_lowresource.sh"
        common = (
            f"--datasets {dataset} --pred-lens {args.pred_lens} --ratios {args.ratios} "
            f"--stage1-epochs {args.stage1_epochs} --train-epochs {args.train_epochs} "
            f"--learning-rate {args.learning_rate:g} --numeric-learning-rate {args.numeric_learning_rate:g} "
            f"--patience {args.patience} --numeric-loss-fn {args.numeric_loss_fn} "
            f"--numeric-val-metric {args.numeric_val_metric} --full-loss-fn {args.full_loss_fn} "
            f"--full-val-metric {args.full_val_metric} --force-batch-size {args.force_batch_size} "
            f"--force-eval-batch-size {args.force_eval_batch_size} --method-profile {args.method_profile} --resume"
        )
        ps1.write_text(
            "\n".join(
                [
                    'param([string]$Python = "python")',
                    "$ErrorActionPreference = 'Stop'",
                    "$Root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)))",
                    "Push-Location $Root",
                    "try {",
                    f"  & $Python tools/run_main_lowresource_grid.py {common}",
                    "}",
                    "finally { Pop-Location }",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        sh.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    'PYTHON="${PYTHON:-python}"',
                    'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"',
                    'cd "$ROOT"',
                    f'"$PYTHON" tools/run_main_lowresource_grid.py {common}',
                    "",
                ]
            ),
            encoding="utf-8",
        )


def write_tables(root: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Path, Path, Path]:
    tables_dir = root / "tables"
    tables_dir.mkdir(exist_ok=True)
    csv_path = tables_dir / "rcare_main_lowresource_results.csv"
    md_path = tables_dir / "rcare_main_lowresource_results.md"
    json_path = root / "outputs" / "rcare_main_lowresource_results.json"
    json_path.parent.mkdir(exist_ok=True)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "features": "M",
            "seq_len": args.seq_len,
            "pred_lens": parse_ints(args.pred_lens),
            "train_ratios": parse_floats(args.ratios),
            "method_profile": args.method_profile,
            "numeric_strategy": "MA7-FRFT warmup_unfreeze",
            "text_source": "deterministic structured summaries cached under generated/<dataset>/",
        },
        "records": records,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "dataset",
        "pred_len",
        "train_ratio",
        "stage1_epochs",
        "train_epochs",
        "batch_size",
        "eval_batch_size",
        "numeric_only_mse",
        "full_student_mse",
        "full_gain",
        "full_gain_pct",
        "full_student_mae",
        "full_teacher_oracle_mse",
        "full_ntr",
        "full_ptr",
        "full_gate",
        "full_reliability",
        "full_robust_mse_avg",
        "stage1_seconds",
        "train_seconds",
        "full_metric_path",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    lines = [
        "# RCARE Main Low-Resource Multivariate Results",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- protocol: multivariate; seq_len={args.seq_len}; pred_len={args.pred_lens}; train ratios={args.ratios}.",
        "- excluded datasets: electricity, traffic, METR-LA, ETTm1, ETTm2, solar_AL.",
        "- text artifacts are cached per dataset under `generated/<dataset>/`.",
        "",
        "| dataset | pred_len | ratio | numeric MSE | student MSE | gain | gain % | student MAE | teacher MSE | NTR | PTR | gate | rel | corrupt avg MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['dataset']} | {row['pred_len']} | {row['train_ratio']:.0%} | "
            f"{row['numeric_only_mse']:.6f} | {row['full_student_mse']:.6f} | "
            f"{row['full_gain']:+.6f} | {row['full_gain_pct']:+.2f}% | "
            f"{row['full_student_mae']:.6f} | {row['full_teacher_oracle_mse']:.6f} | "
            f"{row['full_ntr']:.3f} | {row['full_ptr']:.3f} | {row['full_gate']:.3f} | "
            f"{row['full_reliability']:.3f} | {row['full_robust_mse_avg']:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    results_path = root / "results.txt"
    old = results_path.read_text(encoding="utf-8") if results_path.exists() else ""
    marker = "## 2026-06-08 RCARE main low-resource multivariate grid"
    section = "\n\n" + marker + "\n\n" + "\n".join(lines[2:]) + "\n"
    if marker in old:
        old = old[: old.index(marker)].rstrip()
    results_path.write_text(old + section, encoding="utf-8")
    return csv_path, md_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RCARE main low-resource multivariate grid.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", default="", help="Comma-separated datasets. Empty means every dataset except the skip list.")
    parser.add_argument("--skip", default=",".join(sorted(DEFAULT_SKIP)))
    parser.add_argument("--pred-lens", default="96,192,336,720")
    parser.add_argument("--ratios", default="0.05,0.10,0.20")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--stage1-epochs", type=int, default=8)
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--numeric-learning-rate", type=float, default=5e-5)
    parser.add_argument("--numeric-loss-fn", default="mae")
    parser.add_argument("--numeric-val-metric", default="loss")
    parser.add_argument("--full-loss-fn", default="mae")
    parser.add_argument("--full-val-metric", default="loss")
    parser.add_argument("--force-batch-size", type=int, default=0)
    parser.add_argument("--force-eval-batch-size", type=int, default=0)
    parser.add_argument(
        "--method-profile",
        default="safe_residual",
        choices=["safe_residual", "selective_residual", "semantic_planner", "privileged_bridge"],
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-text", action="store_true")
    parser.add_argument("--write-scripts-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    skip = set(parse_list(args.skip))
    datasets = parse_list(args.datasets) if args.datasets else dataset_names(root, skip)
    pred_lens = parse_ints(args.pred_lens)
    ratios = parse_floats(args.ratios)

    write_dataset_scripts(root, datasets, args)
    master_ps1 = root / "scripts" / "run_rcare_main_lowresource_grid.ps1"
    master_ps1.write_text(
        "\n".join(
            [
                'param([string]$Python = "python")',
                "$ErrorActionPreference = 'Stop'",
                "$Root = Split-Path -Parent $MyInvocation.MyCommand.Path",
                "Push-Location (Split-Path -Parent $Root)",
                "try {",
                f"  & $Python tools/run_main_lowresource_grid.py --datasets {','.join(datasets)} --pred-lens {args.pred_lens} --ratios {args.ratios} --stage1-epochs {args.stage1_epochs} --train-epochs {args.train_epochs} --learning-rate {args.learning_rate:g} --numeric-learning-rate {args.numeric_learning_rate:g} --patience {args.patience} --numeric-loss-fn {args.numeric_loss_fn} --numeric-val-metric {args.numeric_val_metric} --full-loss-fn {args.full_loss_fn} --full-val-metric {args.full_val_metric} --force-batch-size {args.force_batch_size} --force-eval-batch-size {args.force_eval_batch_size} --method-profile {args.method_profile} --resume",
                "}",
                "finally { Pop-Location }",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if args.write_scripts_only:
        print("Wrote per-dataset scripts under scripts/datasets/<dataset>/")
        return 0

    records: list[dict[str, Any]] = []
    for dataset in datasets:
        for pred_len in pred_lens:
            text_path, feature_path = prepare_artifacts(
                root,
                args.python,
                dataset,
                args.seq_len,
                pred_len,
                args.text_dim,
                args.force_text,
                args.dry_run,
            )
            for ratio in ratios:
                bs, ebs = choose_batch_size(dataset, pred_len, root)
                if args.force_batch_size > 0:
                    bs = args.force_batch_size
                if args.force_eval_batch_size > 0:
                    ebs = args.force_eval_batch_size
                stage1_epochs, train_epochs = choose_epochs(pred_len, ratio, args.stage1_epochs, args.train_epochs)
                case = RunCase(
                    dataset=dataset,
                    pred_len=pred_len,
                    train_ratio=ratio,
                    stage1_epochs=stage1_epochs,
                    train_epochs=train_epochs,
                    batch_size=bs,
                    eval_batch_size=ebs,
                    split_mode=split_mode_for(dataset),
                )
                record = run_case(root, args.python, case, text_path, feature_path, args)
                if record:
                    records.append(record)
                    write_tables(root, records, args)

    if args.dry_run:
        return 0
    csv_path, md_path, json_path = write_tables(root, records, args)
    print(csv_path)
    print(md_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


