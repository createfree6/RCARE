from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def infer_dims(args) -> None:
    csv_path = Path(args.root_path) / args.data_path
    df = pd.read_csv(csv_path, nrows=5)
    numeric_cols = [col for col in df.columns if col != "date"]
    if args.target not in numeric_cols:
        raise ValueError(f"Target {args.target!r} not found in {numeric_cols}")
    args.enc_in = len(numeric_cols)
    args.c_out = len(numeric_cols) if args.features == "M" else 1
    args.target_idx = numeric_cols.index(args.target)
    args.numeric_cols = numeric_cols


def normalize_text_args(args) -> None:
    student_cols = getattr(args, "student_text_cols", None)
    if not student_cols:
        student_cols = [args.student_text_col]
    student_cols = [str(col).strip() for col in student_cols if str(col).strip()]
    if not student_cols:
        student_cols = [args.student_text_col]
    args.student_text_cols = student_cols
    args.student_text_col = student_cols[0]
    args.student_text_dim = args.text_dim * len(student_cols)

    teacher_cols = getattr(args, "teacher_text_cols", None)
    if not teacher_cols:
        teacher_cols = [args.teacher_text_col]
    teacher_cols = [str(col).strip() for col in teacher_cols if str(col).strip()]
    if not teacher_cols:
        teacher_cols = [args.teacher_text_col]
    args.teacher_text_cols = teacher_cols
    args.teacher_text_dim = args.text_dim * len(teacher_cols)


def build_setting(args, itr: int) -> str:
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}"
        f"_sl{args.seq_len}_pl{args.pred_len}_hd{args.hidden_dim}_sd{args.sem_dim}_{args.des}_{itr}"
    )


def _cli_has_arg(name: str) -> bool:
    flag = f"--{name}"
    return any(item == flag or item.startswith(f"{flag}=") for item in sys.argv[1:])


def _set_if_unset(args, name: str, value) -> list[str]:
    if _cli_has_arg(name):
        return []
    setattr(args, name, value)
    return [f"{name}={value}"]


def apply_method_profile(args) -> None:
    """Apply paper-oriented presets without overriding explicit CLI values."""
    profile = getattr(args, "method_profile", "manual")
    notes: list[str] = []
    if profile == "manual":
        args.profile_applied = []
        return
    if profile == "numeric_only":
        presets = {
            "ablation": "numeric_only",
            "use_quality_aug": 0,
            "lambda_teacher": 0.0,
            "lambda_base": 0.0,
            "lambda_distill": 0.0,
            "lambda_quality": 0.0,
            "lambda_safety": 0.0,
            "lambda_soft_teacher": 0.0,
            "numeric_backbone": "frft",
            "base_type": "frft",
            "frft_init_alpha": 0.4,
        }
    elif profile == "safe_residual":
        presets = {
            "numeric_backbone": "frft",
            "base_type": "frft",
            "frft_init_alpha": 0.4,
            "adapter_type": "direct",
            "residual_planner_type": "mlp",
            "use_quality_aug": 1,
            "loss_fn": "mae",
            "val_metric": "loss",
            "lambda_teacher": 0.6,
            "lambda_distill": 0.2,
            "distill_pred_weight": 0.5,
            "distill_residual_weight": 0.2,
            "distill_fit_weight": 0.1,
            "lambda_safety": 5.0,
            "lambda_quality": 0.02,
            "reliability_floor": 0.0,
            "reliability_target": "quality_advantage",
            "residual_budget_type": "history_std",
            "residual_budget_scale": 0.5,
            "residual_budget_apply": "student",
            "use_base_context": 0,
        }
    elif profile == "selective_residual":
        presets = {
            "numeric_backbone": "frft",
            "base_type": "frft",
            "frft_init_alpha": 0.4,
            "adapter_type": "direct",
            "residual_planner_type": "mlp",
            "use_quality_aug": 1,
            "loss_fn": "mae",
            "val_metric": "loss",
            "lambda_teacher": 0.6,
            "lambda_distill": 0.18,
            "distill_pred_weight": 0.4,
            "distill_residual_weight": 0.2,
            "distill_fit_weight": 0.1,
            "lambda_residual_aux": 0.03,
            "residual_mag_weight": 0.3,
            "residual_aux_teacher_weight": 0.5,
            "use_residual_aux": 1,
            "use_selective_distill": 1,
            "use_counterfactual_reliability": 1,
            "selective_gain_threshold": 0.0,
            "selective_min_weight": 0.05,
            "lambda_safety": 5.0,
            "lambda_quality": 0.02,
            "reliability_floor": 0.0,
            "reliability_target": "quality_advantage",
            "residual_budget_type": "history_std",
            "residual_budget_scale": 0.5,
            "residual_budget_apply": "student",
            "use_base_context": 0,
        }
    elif profile == "semantic_planner":
        presets = {
            "numeric_backbone": "frft",
            "base_type": "frft",
            "frft_init_alpha": 0.4,
            "adapter_type": "direct",
            "residual_planner_type": "cross_attn",
            "planner_heads": 4,
            "use_quality_aug": 1,
            "loss_fn": "mae",
            "val_metric": "loss",
            "lambda_teacher": 0.6,
            "lambda_distill": 0.2,
            "distill_pred_weight": 0.5,
            "distill_residual_weight": 0.2,
            "distill_fit_weight": 0.1,
            "lambda_safety": 5.0,
            "lambda_quality": 0.02,
            "reliability_floor": 0.0,
            "reliability_target": "quality_advantage",
            "residual_budget_type": "history_std",
            "residual_budget_scale": 0.5,
            "residual_budget_apply": "student",
            "use_base_context": 0,
        }
    elif profile == "single_bridge":
        presets = {
            "numeric_backbone": "frft",
            "base_type": "frft",
            "frft_init_alpha": 0.4,
            "adapter_type": "direct",
            "use_quality_aug": 1,
            "loss_fn": "mae",
            "val_metric": "loss",
            "lambda_teacher": 0.8,
            "lambda_distill": 0.8,
            "distill_pred_weight": 1.0,
            "distill_plan_weight": 0.6,
            "distill_residual_weight": 0.6,
            "distill_gate_weight": 0.2,
            "distill_fit_weight": 0.2,
            "distill_temperature": 0.02,
            "use_selective_distill": 1,
            "selective_min_weight": 0.05,
            "lambda_safety": 2.0,
            "lambda_quality": 0.02,
            "reliability_floor": 0.0,
            "reliability_target": "quality_advantage",
            "residual_budget_type": "history_std",
            "residual_budget_scale": 1.0,
            "residual_budget_apply": "student",
            "use_base_context": 1,
        }
    elif profile == "privileged_bridge":
        presets = {
            "numeric_backbone": "frft",
            "base_type": "frft",
            "frft_init_alpha": 0.4,
            "adapter_type": "direct",
            "residual_planner_type": "mlp",
            "use_base_context": 1,
            "use_quality_aug": 1,
            "loss_fn": "mae",
            "val_metric": "loss",
            "lambda_teacher": 0.6,
            "lambda_distill": 0.25,
            "distill_pred_weight": 0.5,
            "distill_plan_weight": 0.4,
            "distill_residual_weight": 0.3,
            "distill_gate_weight": 0.05,
            "distill_fit_weight": 0.1,
            "lambda_safety": 1.0,
            "lambda_quality": 0.02,
            "reliability_floor": 0.0,
            "reliability_target": "quality_advantage",
            "residual_budget_type": "history_std",
            "residual_budget_scale": 0.75,
            "residual_budget_apply": "student",
        }
    else:
        raise ValueError(f"Unsupported method_profile: {profile}")

    for name, value in presets.items():
        notes.extend(_set_if_unset(args, name, value))
    args.profile_applied = notes


def main() -> int:
    parser = argparse.ArgumentParser(description="CARE-Forecast long-term forecasting")

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="ETTh1_96_96")
    parser.add_argument("--model", type=str, default="CARE_Forecast", choices=["CARE_Forecast", "CARE_S_Forecast"])
    parser.add_argument(
        "--method_profile",
        type=str,
        default="manual",
        choices=[
            "manual",
            "numeric_only",
            "safe_residual",
            "selective_residual",
            "semantic_planner",
            "single_bridge",
            "privileged_bridge",
        ],
        help=(
            "Paper-oriented preset. Explicit CLI arguments always override the preset. "
            "Use safe_residual for the current robust main model; selective_residual "
            "adds selective privileged distillation and residual direction/magnitude heads."
        ),
    )

    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="dataset/ETTh1.csv")
    parser.add_argument("--text_path", type=str, default="generated/ETTh1_sl96_pl96_text_M.csv")
    parser.add_argument("--text_feature_path", type=str, default="", help="Optional NPZ with precomputed text embeddings keyed by text column.")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--split_mode", type=str, default="ett_standard", choices=["ett_standard", "ratio"])
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--target_idx", type=int, default=6)

    parser.add_argument("--text_dim", type=int, default=256)
    parser.add_argument("--text_encoder", type=str, default="hybrid", choices=["hybrid", "hash"])
    parser.add_argument("--student_text_col", type=str, default="compact_text")
    parser.add_argument(
        "--student_text_cols",
        nargs="+",
        default=None,
        help=(
            "Columns concatenated as deployable student-side historical text. "
            "Use multiple history-only views for univariate information enrichment."
        ),
    )
    parser.add_argument(
        "--student_ensemble_text_cols",
        nargs="+",
        default=None,
        help=(
            "History-only text columns used for test-time prompt self-consistency ensembling. "
            "This does not change the model input dimension or use privileged future text."
        ),
    )
    parser.add_argument(
        "--student_ensemble_mode",
        type=str,
        default="mean",
        choices=["mean", "reliability"],
        help="How to combine prompt-ensemble student predictions at evaluation time.",
    )
    parser.add_argument("--teacher_text_col", type=str, default="residual_text")
    parser.add_argument(
        "--teacher_text_cols",
        nargs="+",
        default=None,
        help=(
            "Columns concatenated as privileged teacher-only text. "
            "When omitted, falls back to --teacher_text_col for backward compatibility."
        ),
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sem_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_revin", type=int, default=0)
    parser.add_argument("--numeric_backbone", type=str, default="mlp", choices=["mlp", "frft"])
    parser.add_argument("--base_type", type=str, default="hybrid", choices=["mlp", "nlinear", "hybrid", "frft", "frft_hybrid"])
    parser.add_argument("--frft_patch_len", type=int, default=0, help="0 means output-scale patch length when divisible, otherwise seq_len.")
    parser.add_argument("--frft_init_alpha", type=float, default=0.4, help="Fixed single fractional order for the FRFT numeric branch.")
    parser.add_argument(
        "--decomp_type",
        type=str,
        default="moving_avg",
        choices=["moving_avg", "ema", "dema"],
        help="Trend decomposition; moving_avg is the parallel default, EMA/DEMA are kept for ablation.",
    )
    parser.add_argument("--moving_avg_kernel", type=int, default=7, help="Kernel size for moving-average trend extraction.")
    parser.add_argument("--ema_alpha", type=float, default=0.3, help="EMA smoothing factor for trend extraction.")
    parser.add_argument("--dema_beta", type=float, default=0.3, help="DEMA slope smoothing factor when decomp_type=dema.")
    parser.add_argument("--spectral_mix_init", type=float, default=-1.5, help="Initial logit for spectral residual weight in the numeric backbone.")
    parser.add_argument(
        "--target_only_numeric",
        type=int,
        default=0,
        help="For features=S, feed only the target channel into the numeric backbone instead of all variables.",
    )
    parser.add_argument("--global_mix_init", type=float, default=-10.0, help=argparse.SUPPRESS)
    parser.add_argument("--use_patch_branch", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--patch_branch_len", type=int, default=16)
    parser.add_argument("--patch_branch_stride", type=int, default=8)
    parser.add_argument("--patch_d_model", type=int, default=128)
    parser.add_argument("--patch_layers", type=int, default=2)
    parser.add_argument("--patch_heads", type=int, default=4)
    parser.add_argument("--adapter_type", type=str, default="direct", choices=["basis", "direct"])
    parser.add_argument("--basis_rank", type=int, default=8)
    parser.add_argument(
        "--residual_planner_type",
        type=str,
        default="mlp",
        choices=["mlp", "cross_attn"],
        help="Residual correction decoder: simple MLP or horizon-token semantic cross-attention.",
    )
    parser.add_argument("--planner_heads", type=int, default=4)
    parser.add_argument(
        "--residual_budget_type",
        type=str,
        default="none",
        choices=["none", "history_std", "history_mad"],
        help="Bound semantic residual magnitude by a numeric history scale to reduce negative transfer.",
    )
    parser.add_argument("--residual_budget_scale", type=float, default=1.0)
    parser.add_argument("--residual_budget_min", type=float, default=0.05)
    parser.add_argument("--student_residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--use_base_context",
        type=int,
        default=0,
        help="Append a compact numeric forecast-prior summary to the semantic residual estimator.",
    )
    parser.add_argument(
        "--use_text_modulation",
        type=int,
        default=0,
        help="Force the student residual decoder through history-text FiLM modulation.",
    )
    parser.add_argument("--text_mod_scale", type=float, default=0.2)
    parser.add_argument(
        "--text_mod_mode",
        type=str,
        default="film",
        choices=["film", "gated", "forced_gated"],
        help="How history text modulates the deployable semantic residual.",
    )
    parser.add_argument(
        "--text_mod_zero_init",
        type=int,
        default=1,
        help="Zero-initialize the text modulation head so the initial model is close to the unmodulated residual path.",
    )
    parser.add_argument("--student_numeric_context_scale", type=float, default=1.0)
    parser.add_argument("--student_base_context_scale", type=float, default=1.0)
    parser.add_argument(
        "--residual_budget_apply",
        type=str,
        default="student",
        choices=["student", "both"],
        help="Apply residual budget to the deployable student only, or to both student and teacher residuals.",
    )
    parser.add_argument(
        "--share_residual_decoder",
        type=int,
        default=0,
        help="Share the teacher residual decoder with the student so the predicted residual semantics use the teacher-trained correction head.",
    )
    parser.add_argument(
        "--share_gate_decoder",
        type=int,
        default=0,
        help="Share the teacher gate decoder with the student before applying student reliability.",
    )
    parser.add_argument(
        "--use_pseudo_future_sem",
        type=int,
        default=0,
        help="Append a no-leakage semantic proxy computed from the numeric base forecast to the residual decoders.",
    )
    parser.add_argument("--single_gate_floor", type=float, default=0.0, help="CARE_S_Forecast clean student gate floor.")
    parser.add_argument("--single_teacher_gate_floor", type=float, default=0.0, help="CARE_S_Forecast teacher gate floor.")
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_teacher", "no_gate", "no_quality_aug", "direct_fusion", "numeric_only"],
        help="Controlled ablation mode for paper experiments.",
    )

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--numeric_learning_rate",
        type=float,
        default=0.0,
        help="Optional smaller LR for FRFT numeric backbone during joint fine-tuning; 0 uses learning_rate for all parameters.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--des", type=str, default="formal")
    parser.add_argument("--lradj", type=str, default="constant")
    parser.add_argument("--drop_last", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="Low-resource protocol: keep only this fraction of training windows while preserving validation/test splits.",
    )
    parser.add_argument("--train_ratio_seed", type=int, default=2026)
    parser.add_argument(
        "--train_ratio_mode",
        type=str,
        default="uniform",
        choices=["uniform", "random", "prefix"],
        help="How to subsample training windows when train_ratio < 1.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ntr_eps", type=float, default=1e-6)
    parser.add_argument(
        "--calibrate_residual",
        type=int,
        default=0,
        help="Use validation labels to tune a scalar residual scale and reliability fallback threshold before test evaluation.",
    )
    parser.add_argument(
        "--calibration_scales",
        type=str,
        default="0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3",
    )
    parser.add_argument(
        "--calibration_thresholds",
        type=str,
        default="0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )
    parser.add_argument(
        "--calibration_score",
        type=str,
        default="rel_gate",
        choices=["reliability", "gate", "rel_gate"],
    )
    parser.add_argument("--loss_fn", type=str, default="huber", choices=["mse", "huber", "mae", "huber_mae", "mse_mae"])
    parser.add_argument("--val_metric", type=str, default="loss", choices=["loss", "mse", "mae"], help="Metric used for early stopping.")
    parser.add_argument("--huber_beta", type=float, default=0.5)
    parser.add_argument("--mae_weight", type=float, default=0.5)
    parser.add_argument("--use_quality_aug", type=int, default=1)
    parser.add_argument("--use_residual_aux", type=int, default=0, help="Enable residual direction/magnitude auxiliary heads.")
    parser.add_argument("--use_selective_distill", type=int, default=0, help="Distill privileged teacher mainly when it beats the numeric base.")
    parser.add_argument(
        "--use_counterfactual_reliability",
        type=int,
        default=0,
        help="Use the teacher-vs-base advantage as the clean reliability target.",
    )
    parser.add_argument("--positive_text_cols", nargs="+", default=["history_text", "paraphrase_text", "compact_text"])
    parser.add_argument(
        "--negative_text_cols",
        nargs="+",
        default=["contradictory_text", "noisy_text", "missing_text", "time_shift_text", "irrelevant_text"],
    )

    parser.add_argument("--lambda_teacher", type=float, default=0.8, help="Teacher oracle forecasting supervision.")
    parser.add_argument("--lambda_base", type=float, default=0.2, help="Auxiliary numeric-base forecasting supervision.")
    parser.add_argument("--lambda_distill", type=float, default=0.5, help="Student prediction/semantic/residual distillation from the teacher.")
    parser.add_argument("--lambda_residual_aux", type=float, default=0.0, help="Residual direction/magnitude auxiliary supervision.")
    parser.add_argument("--lambda_soft_teacher", type=float, default=0.0, help="Optional offline oracle-teacher prediction distillation.")
    parser.add_argument("--lambda_quality", type=float, default=0.05, help="Text reliability and noisy-text robustness regularization.")
    parser.add_argument("--lambda_text_contrast", type=float, default=0.0, help="Clean text should outperform shuffled or missing text.")
    parser.add_argument("--text_contrast_margin", type=float, default=0.0)
    parser.add_argument(
        "--use_explicit_text_contrast",
        type=int,
        default=0,
        help="Add explicit shuffled-text and no-text forward passes for text-effect ranking.",
    )
    parser.add_argument("--text_contrast_include_shuffle", type=int, default=1)
    parser.add_argument("--text_contrast_include_no_text", type=int, default=1)
    parser.add_argument("--text_contrast_rank_weight", type=float, default=0.2)
    parser.add_argument("--text_contrast_fallback_weight", type=float, default=0.0)
    parser.add_argument("--evaluate_shuffled_text", type=int, default=1)
    parser.add_argument("--lambda_safety", type=float, default=1.0, help="Negative-transfer safety regularization against the numeric base.")
    parser.add_argument(
        "--lambda_view_pred",
        type=float,
        default=0.0,
        help="Supervise positive history-only text views so test-time prompt ensembles remain deployable.",
    )
    parser.add_argument("--view_consistency_weight", type=float, default=0.2)
    parser.add_argument(
        "--lambda_oracle_residual",
        type=float,
        default=0.0,
        help="Directly fit the deployable student residual to the label-base residual during training.",
    )
    parser.add_argument(
        "--lambda_advantage_transfer",
        type=float,
        default=0.0,
        help="Encourage the student to recover a fraction of the teacher's sample-wise advantage over the numeric base.",
    )
    parser.add_argument("--advantage_transfer_fraction", type=float, default=0.5)
    parser.add_argument("--advantage_transfer_margin", type=float, default=0.0)
    parser.add_argument("--lambda_phys", type=float, default=0.0)
    parser.add_argument(
        "--loss_balance",
        type=str,
        default="none",
        choices=["none", "ema_adaptive"],
        help="Optional bounded EMA-normalized adaptive balancing for auxiliary losses.",
    )
    parser.add_argument("--adaptive_loss_min", type=float, default=0.2)
    parser.add_argument("--adaptive_loss_max", type=float, default=3.0)
    parser.add_argument("--adaptive_loss_ema", type=float, default=0.98)
    parser.add_argument("--adaptive_loss_reg", type=float, default=1e-3)
    parser.add_argument("--distill_pred_weight", type=float, default=1.0)
    parser.add_argument("--distill_plan_weight", type=float, default=0.2)
    parser.add_argument("--distill_residual_weight", type=float, default=0.5)
    parser.add_argument(
        "--distill_pattern_weight",
        type=float,
        default=0.0,
        help="Privileged residual pattern distillation weight inside the distillation objective.",
    )
    parser.add_argument(
        "--distill_pattern_scale_weight",
        type=float,
        default=1.0,
        help="Weight of multi-scale residual trajectory matching in privileged residual KD.",
    )
    parser.add_argument(
        "--distill_pattern_period_weight",
        type=float,
        default=0.25,
        help="Weight of FFT period-distribution matching in privileged residual KD.",
    )
    parser.add_argument("--distill_pattern_scales", type=int, default=2)
    parser.add_argument("--distill_pattern_temperature", type=float, default=0.5)
    parser.add_argument("--distill_pattern_normalize", type=int, default=1)
    parser.add_argument(
        "--distill_coeff_weight",
        type=float,
        default=0.0,
        help="KL distillation between teacher/student residual-basis coefficients when adapter_type=basis.",
    )
    parser.add_argument("--distill_gate_weight", type=float, default=0.1)
    parser.add_argument(
        "--gate_lazy_weight",
        type=float,
        default=0.0,
        help="Asymmetric extra penalty when the student gate is lower than the teacher-guided gate target.",
    )
    parser.add_argument("--distill_fit_weight", type=float, default=0.2)
    parser.add_argument("--selective_gain_threshold", type=float, default=0.0)
    parser.add_argument("--selective_min_weight", type=float, default=0.0)
    parser.add_argument("--residual_direction_eps", type=float, default=0.02)
    parser.add_argument("--residual_mag_weight", type=float, default=0.3)
    parser.add_argument("--residual_aux_teacher_weight", type=float, default=0.5)
    parser.add_argument("--teacher_advantage_floor", type=float, default=0.0)
    parser.add_argument("--student_margin", type=float, default=0.0)
    parser.add_argument("--teacher_margin", type=float, default=0.02)
    parser.add_argument("--reliability_margin", type=float, default=0.2)
    parser.add_argument("--reliability_floor", type=float, default=0.0, help="Lower bound for clean student reliability used in residual gating.")
    parser.add_argument(
        "--reliability_target",
        type=str,
        default="quality",
        choices=["quality", "teacher_advantage", "quality_advantage"],
        help=(
            "Reliability supervision target. quality treats clean text as reliable; "
            "teacher_advantage opens the gate only when the privileged teacher beats the numeric base; "
            "quality_advantage also scales noisy/augmented labels by that advantage."
        ),
    )
    parser.add_argument(
        "--reliability_warmup_epochs",
        type=int,
        default=0,
        help="Bypass reliability in the student gate for the first N epochs so the residual planner learns before reliability calibration.",
    )
    parser.add_argument(
        "--semantic_teacher_warmup_epochs",
        type=int,
        default=0,
        help="Train only the privileged teacher residual path for the first N epochs before student distillation.",
    )
    parser.add_argument(
        "--disable_safety_during_reliability_warmup",
        type=int,
        default=1,
        help="Set lambda_safety to zero while reliability warm-up is active.",
    )
    parser.add_argument("--gate_margin", type=float, default=0.05)
    parser.add_argument("--distill_temperature", type=float, default=0.05)
    parser.add_argument("--contrastive_tau", type=float, default=0.2)

    # Deprecated compatibility switches from the exploratory loss stack. They
    # are parsed so old scripts do not crash, but are intentionally unused.
    for deprecated in [
        "lambda_plan",
        "lambda_contrastive",
        "lambda_reliability",
        "lambda_robust",
        "lambda_consistency",
        "lambda_delta_distill",
        "lambda_gate_distill",
        "lambda_calibration",
        "lambda_student_margin",
        "lambda_teacher_margin",
        "lambda_adv_gate",
    ]:
        parser.add_argument(f"--{deprecated}", type=float, default=0.0, help=argparse.SUPPRESS)

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--deterministic",
        type=int,
        default=0,
        help="Set cuDNN deterministic=True and benchmark=False for stricter reproducibility.",
    )
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--pretrained_numeric_checkpoint", type=str, default="")
    parser.add_argument("--soft_teacher_path", type=str, default="", help="NPZ exported by tools/export_soft_teacher.py with key teacher_pred.")
    parser.add_argument("--freeze_numeric_backbone", type=int, default=0)

    args = parser.parse_args()
    apply_method_profile(args)
    normalize_text_args(args)
    if args.ablation in {"no_quality_aug", "numeric_only"}:
        args.use_quality_aug = 0
    set_seed(args.seed, deterministic=bool(args.deterministic))
    args.use_gpu = bool(args.use_gpu and torch.cuda.is_available())
    infer_dims(args)
    # print("Args:")
    # for key, value in sorted(vars(args).items()):
    #     print(f"  {key}: {value}")

    if args.task_name != "long_term_forecast":
        raise ValueError("Only long_term_forecast is supported.")
    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            setting = build_setting(args, ii)
            exp = Exp(args)
            print(f">>>>>>> start training: {setting} >>>>>>>")
            exp.train(setting)
            print(f">>>>>>> testing: {setting} <<<<<<<")
            exp.test(setting, test=1)
            torch.cuda.empty_cache()
    else:
        setting = build_setting(args, 0)
        exp = Exp(args)
        print(f">>>>>>> testing: {setting} <<<<<<<")
        exp.test(setting, test=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
