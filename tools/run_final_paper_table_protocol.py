from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Only arguments that are actually accepted by run.py are forwarded.
RUN_ARGS = [
    "task_name", "is_training", "model_id", "model", "data", "root_path", "data_path",
    "text_path", "text_feature_path", "features", "target", "freq", "split_mode", "scale",
    "inverse", "seq_len", "label_len", "pred_len", "enc_in", "c_out", "target_idx",
    "text_dim", "text_encoder", "student_text_col", "student_text_cols", "student_ensemble_text_cols",
    "student_ensemble_mode", "teacher_text_col", "teacher_text_cols", "hidden_dim", "sem_dim",
    "dropout", "use_revin", "numeric_backbone", "base_type", "frft_patch_len",
    "frft_init_alpha", "decomp_type", "moving_avg_kernel", "ema_alpha", "dema_beta",
    "spectral_mix_init", "global_mix_init", "use_patch_branch", "patch_branch_len",
    "patch_branch_stride", "patch_d_model", "patch_layers", "patch_heads", "adapter_type",
    "basis_rank", "residual_planner_type", "planner_heads", "residual_budget_type",
    "residual_budget_scale", "residual_budget_min", "student_residual_scale", "residual_budget_apply",
    "use_text_modulation", "text_mod_mode", "text_mod_scale", "text_mod_zero_init",
    "use_base_context", "student_numeric_context_scale", "student_base_context_scale",
    "share_residual_decoder", "share_gate_decoder", "single_gate_floor", "single_teacher_gate_floor",
    "num_workers", "itr", "train_epochs", "batch_size", "eval_batch_size", "patience",
    "learning_rate", "numeric_learning_rate", "weight_decay", "des", "lradj", "drop_last",
    "max_samples", "train_ratio", "train_ratio_seed", "train_ratio_mode", "max_grad_norm",
    "ntr_eps", "method_profile", "loss_fn", "val_metric", "huber_beta", "mae_weight",
    "use_quality_aug", "use_residual_aux", "use_selective_distill", "positive_text_cols",
    "negative_text_cols", "lambda_teacher", "lambda_base", "lambda_distill", "lambda_residual_aux",
    "lambda_soft_teacher", "lambda_quality", "lambda_text_contrast", "text_contrast_margin",
    "text_contrast_include_shuffle", "text_contrast_include_no_text", "text_contrast_rank_weight",
    "text_contrast_fallback_weight", "evaluate_shuffled_text", "lambda_safety", "lambda_consistency",
    "view_consistency_weight", "lambda_view_pred", "lambda_advantage_transfer",
    "advantage_transfer_fraction", "advantage_transfer_margin", "lambda_phys", "loss_balance",
    "adaptive_loss_min", "adaptive_loss_max", "adaptive_loss_ema", "adaptive_loss_reg",
    "distill_pred_weight", "distill_plan_weight", "distill_residual_weight", "distill_coeff_weight",
    "distill_pattern_weight", "distill_pattern_scale_weight", "distill_pattern_period_weight",
    "distill_pattern_scales", "distill_pattern_temperature", "distill_pattern_normalize",
    "distill_gate_weight", "distill_fit_weight", "selective_gain_threshold", "selective_min_weight",
    "residual_direction_eps", "residual_mag_weight", "residual_aux_teacher_weight",
    "teacher_advantage_floor", "student_margin", "teacher_margin", "reliability_margin",
    "reliability_floor", "reliability_target", "reliability_warmup_epochs",
    "disable_safety_during_reliability_warmup", "use_counterfactual_reliability",
    "calibrate_residual", "calibration_scales", "calibration_thresholds", "calibration_score",
    "gate_margin", "distill_temperature", "contrastive_tau", "lambda_plan", "lambda_calibration",
    "lambda_gate_distill", "lambda_oracle_residual", "lambda_adv_gate", "lambda_student_margin",
    "lambda_teacher_margin", "lambda_delta_distill", "use_gpu", "gpu", "seed", "deterministic",
    "checkpoints", "output_dir", "pretrained_numeric_checkpoint", "soft_teacher_path", "freeze_numeric_backbone",
]

STORE_TRUE_ARGS = {"inverse", "drop_last"}
LIST_ARGS = {"student_text_cols", "student_ensemble_text_cols", "teacher_text_cols", "positive_text_cols", "negative_text_cols"}

SUMMARY_KEYS = [
    "model", "data", "seq_len", "label_len", "pred_len", "train_ratio", "train_ratio_seed",
    "batch_size", "eval_batch_size", "train_epochs", "patience", "learning_rate",
    "numeric_learning_rate", "dropout", "hidden_dim", "sem_dim", "numeric_backbone",
    "base_type", "moving_avg_kernel", "frft_init_alpha", "method_profile",
    "freeze_numeric_backbone", "student_text_cols", "teacher_text_cols",
    "residual_budget_scale", "student_residual_scale", "lambda_teacher",
    "lambda_distill", "lambda_safety", "lambda_advantage_transfer",
    "distill_temperature", "seed", "deterministic", "text_path", "text_feature_path",
    "pretrained_numeric_checkpoint", "output_dir",
]


def parse_filter(text: str, cast=str) -> set[Any] | None:
    if not text:
        return None
    return {cast(x.strip()) for x in text.split(",") if x.strip()}


def load_metric(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metric json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def value_to_tokens(key: str, value: Any) -> list[str]:
    if value is None:
        return []
    if key in STORE_TRUE_ARGS:
        return [f"--{key}"] if bool(value) else []
    if key in LIST_ARGS:
        if isinstance(value, str):
            parts = [p for p in value.replace(",", " ").split() if p]
        else:
            parts = [str(p) for p in value]
        return [f"--{key}", *parts] if parts else []
    if isinstance(value, bool):
        return [f"--{key}", "True" if value else "False"]
    if isinstance(value, (list, tuple)):
        return [f"--{key}", *[str(v) for v in value]] if value else []
    return [f"--{key}", str(value)]


def command_from_metric(metric: dict[str, Any], python: str) -> list[str]:
    args = metric.get("args")
    if not isinstance(args, dict):
        raise ValueError("metrics.json does not contain an args dictionary")
    cmd = [python, "run.py"]
    for key in RUN_ARGS:
        if key in args:
            cmd.extend(value_to_tokens(key, args[key]))
    return cmd


def print_param_summary(metric: dict[str, Any]) -> None:
    args = metric.get("args")
    if not isinstance(args, dict):
        print("Parameter summary: unavailable because metrics.json has no args dictionary")
        return
    print("Parameter summary from metrics.json args:")
    for key in SUMMARY_KEYS:
        if key in args and args[key] not in (None, "", []):
            print(f"  {key}: {args[key]}")


def checkpoint_from_setting(setting: str, root: Path) -> Path:
    return root / "checkpoints" / setting / "checkpoint.pth"


def run_command(cmd: list[str], root: Path, dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, cwd=root, env=env, check=True)


def select_rows(args: argparse.Namespace, root: Path) -> list[dict[str, str]]:
    datasets = parse_filter(args.datasets, str)
    pred_lens = parse_filter(args.pred_lens, int)
    ratios = parse_filter(args.ratios, float)
    with (root / args.manifest_csv).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected = []
    for row in rows:
        if datasets and row["dataset"] not in datasets:
            continue
        if pred_lens and int(float(row["pred_len"])) not in pred_lens:
            continue
        if ratios and round(float(row["train_ratio"]), 6) not in {round(x, 6) for x in ratios}:
            continue
        selected.append(row)
    return selected


def final_source_path(row: dict[str, str], root: Path) -> tuple[str, Path]:
    action = row.get("positive_safe_action", "")
    if action.startswith("numeric_fallback"):
        return "numeric_fallback", root / row["numeric_metric_path"]
    return "semantic_or_tuned", root / row["full_metric_path"]


def maybe_run_numeric(row: dict[str, str], root: Path, python: str, force: bool, dry_run: bool) -> None:
    numeric_metric_path = root / row["numeric_metric_path"]
    numeric_setting = row.get("numeric_setting", "")
    ckpt = checkpoint_from_setting(numeric_setting, root)
    if ckpt.exists() and not force:
        return
    metric = load_metric(numeric_metric_path)
    print(f"Ensuring numeric checkpoint for {row['dataset']} H={row['pred_len']} r={row['train_ratio']} -> {rel(ckpt, root)}")
    print_param_summary(metric)
    run_command(command_from_metric(metric, python), root, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run/verify the final paper-table RCARE protocol from semantic_v1_positive_safe_significant_results.csv.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--manifest-csv", default="tables/semantic_v1_positive_safe_significant_results.csv")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names, e.g. Wind,ZafNoo. Empty means all.")
    parser.add_argument("--pred-lens", default="", help="Comma-separated horizons, e.g. 96,192. Empty means all.")
    parser.add_argument("--ratios", default="", help="Comma-separated train ratios, e.g. 0.1 or 0.05,0.10. Empty means all.")
    parser.add_argument("--force", action="store_true", help="Rerun even if the selected final metrics already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running training.")
    parser.add_argument("--skip-numeric", action="store_true", help="Do not recreate missing numeric checkpoints before full runs.")
    args = parser.parse_args()

    root = Path.cwd()
    rows = select_rows(args, root)
    print(f"Selected {len(rows)} final-table case(s) from {args.manifest_csv}")
    if not rows:
        return 0

    for idx, row in enumerate(rows, 1):
        source_kind, source_metric_path = final_source_path(row, root)
        label = f"[{idx}/{len(rows)}] {row['dataset']} H={row['pred_len']} ratio={row['train_ratio']} action={row.get('positive_safe_action','')} candidate={row.get('positive_safe_candidate','')}"
        print("\n" + "=" * 100)
        print(label)
        print(f"Final source: {source_kind} -> {rel(source_metric_path, root)}")
        if source_metric_path.exists() and not args.force:
            print("Resume: final source metrics already exist. Use -Force to rerun.")
            continue
        if source_kind != "numeric_fallback" and not args.skip_numeric:
            maybe_run_numeric(row, root, args.python, args.force, args.dry_run)
        metric = load_metric(source_metric_path)
        print_param_summary(metric)
        run_command(command_from_metric(metric, args.python), root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
