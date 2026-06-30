from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data_provider.data_factory import data_provider
from data_provider.data_loader import UNRELIABLE_COLUMNS
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


TEXT_QUALITY = {
    "history_text": 1.0,
    "llm_history_text": 1.0,
    "llm_history_future_text": 1.0,
    "paraphrase_text": 0.95,
    "compact_text": 0.85,
    "future_text": 1.0,
    "residual_text": 1.0,
    "llm_future_text": 1.0,
    "llm_residual_text": 1.0,
    "noisy_text": 0.25,
    "time_shift_text": 0.20,
    "contradictory_text": 0.0,
    "missing_text": 0.0,
    "irrelevant_text": 0.0,
}


class ForecastLoss(nn.Module):
    """Regression loss variants for forecasting while keeping validation consistent."""

    def __init__(self, mode: str, huber_beta: float = 0.5, mae_weight: float = 0.5):
        super().__init__()
        self.mode = mode
        self.huber_beta = huber_beta
        self.mae_weight = mae_weight

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if self.mode == "mse":
            return F.mse_loss(pred, true)
        if self.mode == "mae":
            return F.l1_loss(pred, true)
        if self.mode == "huber":
            return F.smooth_l1_loss(pred, true, beta=self.huber_beta)
        if self.mode == "huber_mae":
            return F.smooth_l1_loss(pred, true, beta=self.huber_beta) + self.mae_weight * F.l1_loss(pred, true)
        if self.mode == "mse_mae":
            return F.mse_loss(pred, true) + self.mae_weight * F.l1_loss(pred, true)
        raise ValueError(f"Unsupported loss_fn: {self.mode}")


class AdaptiveAuxLossBalancer(nn.Module):
    """Bounded EMA-normalized balancing for auxiliary objectives.

    The main forecasting loss remains unweighted. This module only learns small
    multiplicative adjustments around user-provided auxiliary lambda priors.
    """

    def __init__(
        self,
        names: list[str],
        min_scale: float = 0.2,
        max_scale: float = 3.0,
        ema_decay: float = 0.98,
        reg: float = 1e-3,
    ):
        super().__init__()
        if not 0.0 < min_scale < 1.0 < max_scale:
            raise ValueError("adaptive loss scales must satisfy 0 < min < 1 < max.")
        self.names = names
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.ema_decay = float(ema_decay)
        self.reg = float(reg)
        init_prob = (1.0 - self.min_scale) / (self.max_scale - self.min_scale)
        init_logit = torch.logit(torch.tensor(init_prob, dtype=torch.float32))
        self.logits = nn.ParameterDict({name: nn.Parameter(init_logit.clone()) for name in names})
        self.register_buffer("ema", torch.ones(len(names), dtype=torch.float32))
        self.register_buffer("ref", torch.ones(len(names), dtype=torch.float32))
        self.register_buffer("initialized", torch.zeros(len(names), dtype=torch.bool))

    def scales(self) -> dict[str, torch.Tensor]:
        return {
            name: self.min_scale + (self.max_scale - self.min_scale) * torch.sigmoid(self.logits[name])
            for name in self.names
        }

    def forward(
        self,
        losses: dict[str, torch.Tensor],
        lambda_priors: dict[str, float],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        first_loss = next(iter(losses.values()))
        total = first_loss.new_tensor(0.0)
        reg_loss = first_loss.new_tensor(0.0)
        logs: dict[str, float] = {}
        scale_map = self.scales()

        for idx, name in enumerate(self.names):
            raw_loss = losses.get(name, first_loss.new_tensor(0.0))
            prior = float(lambda_priors.get(name, 0.0))
            scale = scale_map[name]
            logs[f"adaptive_scale_{name}"] = float(scale.detach().cpu().item())
            logs[f"adaptive_lambda_{name}"] = float(prior * scale.detach().cpu().item())
            if prior <= 0:
                continue

            detached = raw_loss.detach().abs().clamp_min(1e-8)
            if bool(self.initialized[idx].item()):
                self.ema[idx] = self.ema_decay * self.ema[idx] + (1.0 - self.ema_decay) * detached.to(self.ema.device)
            else:
                self.ema[idx] = detached.to(self.ema.device)
                self.ref[idx] = detached.to(self.ref.device)
                self.initialized[idx] = True

            ema = self.ema[idx].to(device=raw_loss.device, dtype=raw_loss.dtype).clamp_min(1e-8)
            ref = self.ref[idx].to(device=raw_loss.device, dtype=raw_loss.dtype).clamp_min(1e-8)
            normalized_loss = raw_loss / ema.detach() * ref.detach()
            total = total + prior * scale * normalized_loss
            reg_loss = reg_loss + torch.log(scale).pow(2)

        if self.reg > 0:
            total = total + self.reg * reg_loss
        logs["adaptive_reg"] = float((self.reg * reg_loss).detach().cpu().item()) if self.reg > 0 else 0.0
        return total, logs


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.loss_balancer = None
        if getattr(args, "loss_balance", "none") == "ema_adaptive":
            self.loss_balancer = AdaptiveAuxLossBalancer(
                names=["teacher", "distill", "soft_teacher", "quality", "safety"],
                min_scale=float(getattr(args, "adaptive_loss_min", 0.2)),
                max_scale=float(getattr(args, "adaptive_loss_max", 3.0)),
                ema_decay=float(getattr(args, "adaptive_loss_ema", 0.98)),
                reg=float(getattr(args, "adaptive_loss_reg", 1e-3)),
            ).to(self.device)

    def _get_data(self, flag: str):
        return data_provider(self.args, flag)




    def _numeric_backbone_info(self) -> dict[str, object] | None:
        frft_backbone = getattr(self.model, "frft_backbone", None)
        if frft_backbone is None:
            return None
        frft_block = frft_backbone.trend_frft
        decomp_module = frft_backbone.decomp.ma
        ema_alpha = getattr(decomp_module, "alpha", None)
        dema_beta = getattr(decomp_module, "beta", None)
        return {
            "type": "movingavg_fft_frft",
            "target_only_numeric": bool(getattr(self.args, "target_only_numeric", 0)),
            "decomposition": frft_backbone.decomp.decomp_type,
            "moving_avg_kernel": int(getattr(decomp_module, "kernel_size", 0)),
            "ema_alpha": float(ema_alpha.detach().cpu().item()) if ema_alpha is not None else 0.0,
            "dema_beta": float(dema_beta.detach().cpu().item()) if dema_beta is not None else 0.0,
            "patch_len": int(frft_block.patch_len),
            "alpha": float(frft_block.current_alpha().detach().cpu().item()),
            "spectral_mix": float(torch.sigmoid(frft_backbone.spectral_mix_logit).detach().cpu().item()),
            "phys_loss_weights": {"ratio": 0.01, "imag_smooth": 0.1, "imag_abs": 0.1},
        }

    def _print_numeric_backbone_info(self, prefix: str) -> None:
        info = self._numeric_backbone_info()
        if info is None:
            return
        print(
            f"{prefix} numeric={info['type']}, decomp={info['decomposition']}, "
            f"target_only={info['target_only_numeric']}, "
            f"moving_avg_kernel={info['moving_avg_kernel']}, ema_alpha={info['ema_alpha']:.4f}, "
            f"frft_alpha={info['alpha']:.6f}, "
            f"patch_len={info['patch_len']}, spectral_mix={info['spectral_mix']:.4f}, "
            f"phys_loss_weights={info['phys_loss_weights']}"
        )

    def _load_pretrained_numeric_checkpoint(self) -> None:
        ckpt = getattr(self.args, "pretrained_numeric_checkpoint", "")
        if not ckpt:
            return
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        try:
            source = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        except TypeError:
            source = torch.load(ckpt_path, map_location=self.device)

        target = self.model.state_dict()
        prefixes = (
            "frft_backbone.",
            "numeric_encoder.",
            "base_head.",
            "channel_linear.",
        )
        scalar_numeric_keys = {"base_mix_logit", "frft_mix_logit"}
        loaded: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, value in source.items():
            if not key.startswith(prefixes) and key not in scalar_numeric_keys:
                continue
            if key not in target or target[key].shape != value.shape:
                skipped.append(key)
                continue
            loaded[key] = value
        target.update(loaded)
        self.model.load_state_dict(target)
        print(f"Loaded {len(loaded)} numeric/base tensors from {ckpt_path}")
        if skipped:
            print(f"Skipped {len(skipped)} numeric tensors due to shape/key mismatch")
        self._print_numeric_backbone_info("After loading pretrained numeric")

    def _freeze_numeric_backbone(self) -> None:
        if not bool(getattr(self.args, "freeze_numeric_backbone", 0)):
            return
        frft_backbone = getattr(self.model, "frft_backbone", None)
        if frft_backbone is None:
            print("freeze_numeric_backbone=1 ignored because no FRFT backbone is active.")
            return
        numeric_modules = [
            frft_backbone,
            getattr(self.model, "numeric_encoder", None),
            getattr(self.model, "base_head", None),
            getattr(self.model, "channel_linear", None),
        ]
        frozen = 0
        for module in numeric_modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
                frozen += param.numel()
        for name in ["base_mix_logit", "frft_mix_logit"]:
            param = getattr(self.model, name, None)
            if isinstance(param, torch.nn.Parameter):
                param.requires_grad = False
                frozen += param.numel()
        print(f"Frozen numeric/base parameters: {frozen}")
        self._print_numeric_backbone_info("Frozen numeric")

    def _keep_frozen_numeric_eval(self) -> None:
        if not bool(getattr(self.args, "freeze_numeric_backbone", 0)):
            return
        frft_backbone = getattr(self.model, "frft_backbone", None)
        if frft_backbone is not None:
            frft_backbone.eval()

    def _set_reliability_warmup(self, active: bool) -> None:
        setattr(self.model, "runtime_reliability_bypass", bool(active))

    def _select_optimizer(self):
        numeric_lr = float(getattr(self.args, "numeric_learning_rate", 0.0) or 0.0)
        frft_backbone = getattr(self.model, "frft_backbone", None)
        balancer_params = (
            [param for param in self.loss_balancer.parameters() if param.requires_grad]
            if self.loss_balancer is not None
            else []
        )
        if numeric_lr > 0 and frft_backbone is not None and not bool(getattr(self.args, "freeze_numeric_backbone", 0)):
            numeric_ids = {id(param) for param in frft_backbone.parameters() if param.requires_grad}
            numeric_params = [param for param in frft_backbone.parameters() if param.requires_grad]
            other_params = [
                param for param in self.model.parameters() if param.requires_grad and id(param) not in numeric_ids
            ]
            param_groups = []
            if numeric_params:
                param_groups.append({"params": numeric_params, "lr": numeric_lr, "name": "numeric_backbone"})
            if other_params:
                param_groups.append({"params": other_params, "lr": self.args.learning_rate, "name": "multimodal"})
            if balancer_params:
                param_groups.append({"params": balancer_params, "lr": self.args.learning_rate, "name": "loss_balancer"})
            print(
                f"Using differential LR: numeric_backbone={numeric_lr:g}, "
                f"multimodal={self.args.learning_rate:g}"
            )
            return torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        params = [param for param in self.model.parameters() if param.requires_grad]
        params.extend(balancer_params)
        return torch.optim.AdamW(params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    def _select_criterion(self):
        return ForecastLoss(
            mode=getattr(self.args, "loss_fn", "mse"),
            huber_beta=self.args.huber_beta,
            mae_weight=self.args.mae_weight,
        )

    def _teacher_enabled(self) -> bool:
        return self.args.ablation not in {"no_teacher", "numeric_only"}

    def _quality_enabled(self) -> bool:
        return bool(self.args.use_quality_aug) and self.args.ablation not in {"no_quality_aug", "numeric_only"}

    def _load_soft_teacher(self) -> np.ndarray | None:
        path_value = getattr(self.args, "soft_teacher_path", "")
        if not path_value:
            return None
        path = Path(path_value)
        if not path.is_absolute():
            path = Path.cwd() / path
        arrays = np.load(path)
        if "teacher_pred" not in arrays:
            raise ValueError(f"soft teacher file {path} must contain key 'teacher_pred'.")
        teacher_pred = arrays["teacher_pred"].astype(np.float32)
        if teacher_pred.ndim != 3 or teacher_pred.shape[1:] != (self.args.pred_len, self.args.c_out):
            raise ValueError(
                f"teacher_pred shape {teacher_pred.shape} does not match (*, {self.args.pred_len}, {self.args.c_out})."
            )
        print(f"Loaded offline soft teacher predictions: {path} {teacher_pred.shape}")
        return teacher_pred

    @staticmethod
    def _arg_columns(value, fallback: list[str]) -> list[str]:
        if value is None:
            return fallback
        if isinstance(value, str):
            cols = [item.strip() for item in value.split(",") if item.strip()]
            return cols or fallback
        return [str(item) for item in value] or fallback

    @staticmethod
    def _available_columns(dataset, columns: list[str]) -> list[str]:
        return [col for col in columns if col in dataset.text_features]

    def _student_override_batch(self, dataset, row_idx: torch.Tensor, col: str, device: torch.device) -> torch.Tensor:
        idx_np = row_idx.detach().cpu().numpy()
        repeat = max(1, len(getattr(dataset, "student_text_cols", [self.args.student_text_col])))
        parts = [dataset.text_features[col][idx_np] for _ in range(repeat)]
        features = parts[0] if repeat == 1 else np.concatenate(parts, axis=-1)
        return torch.from_numpy(features).float().to(device)

    def _sample_text_batch(self, dataset, row_idx: torch.Tensor, columns: list[str], device: torch.device) -> tuple[torch.Tensor, str, float]:
        available = self._available_columns(dataset, columns)
        if not available:
            available = [self.args.student_text_col]
        col = random.choice(available)
        features = self._student_override_batch(dataset, row_idx, col, device)
        return features, col, float(TEXT_QUALITY.get(col, 1.0))

    def _apply_prompt_ensemble(
        self,
        base_out: dict[str, torch.Tensor],
        batch_x: torch.Tensor,
        history_text: torch.Tensor,
        dataset,
        row_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cols = self._arg_columns(getattr(self.args, "student_ensemble_text_cols", None), [])
        cols = self._available_columns(dataset, cols)
        if not cols:
            return base_out

        seen: set[str] = set()
        unique_cols: list[str] = []
        for col in cols:
            if col not in seen:
                seen.add(col)
                unique_cols.append(col)
        if not unique_cols:
            return base_out

        candidates: list[dict[str, torch.Tensor]] = []
        default_cols = set(getattr(dataset, "student_text_cols", [self.args.student_text_col]))
        for col in unique_cols:
            if col in default_cols and len(default_cols) == 1:
                candidates.append(base_out)
            else:
                override = self._student_override_batch(dataset, row_idx, col, self.device)
                candidates.append(self.model(batch_x, history_text, None, override_history_text=override))

        if len(candidates) == 1:
            out = dict(base_out)
            out["student"] = candidates[0]["student"]
            out["student_gate"] = candidates[0]["student_gate"]
            out["student_reliability"] = candidates[0]["student_reliability"]
            out["prompt_ensemble_size"] = candidates[0]["student"].new_tensor(1.0)
            return out

        pred_stack = torch.stack([item["student"] for item in candidates], dim=0)
        gate_stack = torch.stack([item["student_gate"] for item in candidates], dim=0)
        rel_stack = torch.stack([item["student_reliability"] for item in candidates], dim=0)
        mode = getattr(self.args, "student_ensemble_mode", "mean")
        if mode == "reliability":
            weight = rel_stack.mean(dim=-1, keepdim=True).clamp_min(1e-4)
            weight = weight / weight.sum(dim=0, keepdim=True).clamp_min(1e-6)
            pred = (pred_stack * weight.unsqueeze(-1)).sum(dim=0)
            gate = (gate_stack * weight.unsqueeze(-1)).sum(dim=0)
            reliability = (rel_stack * weight).sum(dim=0)
        else:
            pred = pred_stack.mean(dim=0)
            gate = gate_stack.mean(dim=0)
            reliability = rel_stack.mean(dim=0)

        out = dict(base_out)
        out["student"] = pred
        out["student_gate"] = gate
        out["student_reliability"] = reliability
        out["prompt_ensemble_size"] = pred.new_tensor(float(len(candidates)))
        return out

    @staticmethod
    def _negative_transfer_rate(pred: np.ndarray, base: np.ndarray, true: np.ndarray, eps: float = 1e-6) -> float:
        pred_mse = np.mean((pred - true) ** 2, axis=(1, 2))
        base_mse = np.mean((base - true) ** 2, axis=(1, 2))
        return float(np.mean(pred_mse > base_mse + eps))

    @staticmethod
    def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float64).reshape(-1)
        right = np.asarray(right, dtype=np.float64).reshape(-1)
        if left.size == 0 or right.size == 0 or left.size != right.size:
            return 0.0
        left = left - left.mean()
        right = right - right.mean()
        denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
        if denom < 1e-12:
            return 0.0
        return float(np.sum(left * right) / denom)

    @staticmethod
    def _transfer_diagnostics(
        pred: np.ndarray,
        base: np.ndarray,
        true: np.ndarray,
        gate: np.ndarray,
        reliability: np.ndarray,
        eps: float,
    ) -> dict[str, float]:
        pred_mse = np.mean((pred - true) ** 2, axis=(1, 2))
        base_mse = np.mean((base - true) ** 2, axis=(1, 2))
        sample_gain = base_mse - pred_mse
        negative_delta = np.maximum(-sample_gain, 0.0)
        gate_sample = np.mean(gate, axis=(1, 2))
        reliability_sample = reliability.reshape(reliability.shape[0], -1).mean(axis=1)
        return {
            "student_mean_sample_gain": float(np.mean(sample_gain)),
            "student_positive_transfer_rate": float(np.mean(sample_gain > eps)),
            "student_negative_transfer_severity": float(np.mean(negative_delta)),
            "student_negative_transfer_severity_active": float(np.mean(negative_delta[sample_gain < -eps]))
            if np.any(sample_gain < -eps)
            else 0.0,
            "student_reliability_gain_corr": Exp_Long_Term_Forecast._safe_pearson(reliability_sample, sample_gain),
            "student_gate_gain_corr": Exp_Long_Term_Forecast._safe_pearson(gate_sample, sample_gain),
        }

    @staticmethod
    def _parse_float_grid(text: str) -> list[float]:
        values: list[float] = []
        for item in str(text).split(","):
            item = item.strip()
            if item:
                values.append(float(item))
        return values or [0.0, 1.0]

    @staticmethod
    def _calibration_score(gate: np.ndarray, reliability: np.ndarray, mode: str) -> np.ndarray:
        gate_sample = np.mean(gate, axis=(1, 2))
        reliability_sample = reliability.reshape(reliability.shape[0], -1).mean(axis=1)
        if mode == "reliability":
            return reliability_sample
        if mode == "gate":
            return gate_sample
        return reliability_sample * gate_sample

    @staticmethod
    def _apply_residual_calibration(
        pred: np.ndarray,
        base: np.ndarray,
        gate: np.ndarray,
        reliability: np.ndarray,
        calibration: dict[str, float] | None,
    ) -> np.ndarray:
        if not calibration:
            return pred
        score = Exp_Long_Term_Forecast._calibration_score(gate, reliability, str(calibration.get("score_mode", "rel_gate")))
        mask = (score >= float(calibration.get("threshold", 0.0))).astype(pred.dtype).reshape(-1, 1, 1)
        scale = float(calibration.get("scale", 1.0))
        return base + scale * mask * (pred - base)

    @torch.no_grad()
    def _collect_student_arrays(self, dataset, loader, noise_col: str | None = None) -> tuple[np.ndarray, ...]:
        self.model.eval()
        preds, trues, bases, gates, reliabilities = [], [], [], [], []
        for batch_x, batch_y, history_text, teacher_text, _noise_text, row_idx in loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            history_text = history_text.float().to(self.device)
            teacher_text = teacher_text.float().to(self.device)
            if noise_col is None:
                out = self.model(batch_x, history_text, teacher_text if self._teacher_enabled() else None)
                out = self._apply_prompt_ensemble(out, batch_x, history_text, dataset, row_idx)
            else:
                override = self._student_override_batch(dataset, row_idx, noise_col, self.device)
                out = self.model(
                    batch_x,
                    history_text,
                    teacher_text if self._teacher_enabled() else None,
                    override_history_text=override,
                )
            preds.append(out["student"].detach().cpu().numpy())
            bases.append(out["base"].detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            gates.append(out["student_gate"].detach().cpu().numpy())
            reliabilities.append(out["student_reliability"].detach().cpu().numpy())
        return (
            np.concatenate(preds, axis=0),
            np.concatenate(trues, axis=0),
            np.concatenate(bases, axis=0),
            np.concatenate(gates, axis=0),
            np.concatenate(reliabilities, axis=0),
        )

    @torch.no_grad()
    def _calibrate_residual_on_validation(self, vali_data, vali_loader) -> dict[str, float]:
        pred, true, base, gate, reliability = self._collect_student_arrays(vali_data, vali_loader)
        scales = self._parse_float_grid(getattr(self.args, "calibration_scales", "0,1"))
        thresholds = self._parse_float_grid(getattr(self.args, "calibration_thresholds", "0"))
        score_mode = str(getattr(self.args, "calibration_score", "rel_gate"))
        score = self._calibration_score(gate, reliability, score_mode)
        best = {
            "scale": 1.0,
            "threshold": 0.0,
            "score_mode": score_mode,
            "val_mse": float(np.mean((pred - true) ** 2)),
            "val_base_mse": float(np.mean((base - true) ** 2)),
        }
        best_loss = best["val_mse"]
        for scale in scales:
            for threshold in thresholds:
                mask = (score >= threshold).astype(pred.dtype).reshape(-1, 1, 1)
                cal_pred = base + float(scale) * mask * (pred - base)
                loss = float(np.mean((cal_pred - true) ** 2))
                if loss < best_loss - 1e-12:
                    best_loss = loss
                    best.update({"scale": float(scale), "threshold": float(threshold), "val_mse": loss})
        return best

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        while weights.dim() < values.dim():
            weights = weights.unsqueeze(-1)
        weights = weights.expand_as(values)
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _sample_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Average per-sample losses with a selective teacher-advantage weight."""
        return (values * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _time_scales(x: torch.Tensor, levels: int) -> list[torch.Tensor]:
        """Build coarse residual trajectories without introducing extra model parameters."""
        scales = [x]
        cur = x
        for _ in range(max(0, int(levels))):
            if cur.shape[1] < 2:
                break
            cur = F.avg_pool1d(cur.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            scales.append(cur)
        return scales

    @staticmethod
    def _rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        scale = torch.sqrt(torch.mean(x.pow(2), dim=(1, 2), keepdim=True) + eps)
        return x / scale

    def _privileged_residual_pattern_loss(
        self,
        student_residual: torch.Tensor,
        teacher_residual: torch.Tensor,
        distill_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Distill the privileged residual as learnable temporal/frequency patterns.

        This follows the spirit of time-series-aware KD, but applies it only to
        the semantic residual correction. The numeric base itself is not forced
        to imitate the privileged teacher.
        """
        zero = student_residual.new_tensor(0.0)
        if float(getattr(self.args, "distill_pattern_weight", 0.0)) <= 0:
            return zero, zero, zero

        num_scales = int(getattr(self.args, "distill_pattern_scales", 2))
        normalize = bool(getattr(self.args, "distill_pattern_normalize", 1))
        scale_loss = zero
        student_scales = self._time_scales(student_residual, num_scales)
        teacher_scales = self._time_scales(teacher_residual.detach(), num_scales)
        for student_scale, teacher_scale in zip(student_scales, teacher_scales):
            if normalize:
                student_scale = self._rms_normalize(student_scale)
                teacher_scale = self._rms_normalize(teacher_scale)
            per_sample = (student_scale - teacher_scale).pow(2).mean(dim=(1, 2))
            scale_loss = scale_loss + self._sample_weighted_mean(per_sample, distill_weight)
        scale_loss = scale_loss / max(1, len(student_scales))

        period_loss = zero
        if student_residual.shape[1] >= 4 and float(getattr(self.args, "distill_pattern_period_weight", 0.0)) > 0:
            temperature = max(float(getattr(self.args, "distill_pattern_temperature", 0.5)), 1e-4)
            student_mag = torch.abs(torch.fft.rfft(student_residual, dim=1))
            teacher_mag = torch.abs(torch.fft.rfft(teacher_residual.detach(), dim=1))
            if student_mag.shape[1] > 1:
                # Drop the DC bin: mean residual shift is already covered by residual MSE.
                student_log_prob = F.log_softmax(student_mag[:, 1:, :] / temperature, dim=1)
                teacher_prob = F.softmax(teacher_mag[:, 1:, :] / temperature, dim=1)
                per_sample = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1).mean(dim=1)
                period_loss = self._sample_weighted_mean(per_sample, distill_weight)

        total = (
            float(getattr(self.args, "distill_pattern_scale_weight", 1.0)) * scale_loss
            + float(getattr(self.args, "distill_pattern_period_weight", 0.0)) * period_loss
        )
        return total, scale_loss, period_loss

    def _residual_aux_loss(
        self,
        out: dict[str, torch.Tensor],
        y: torch.Tensor,
        distill_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = out["student"].new_tensor(0.0)
        if "student_res_dir_logits" not in out or self.args.lambda_residual_aux <= 0:
            return zero, zero, zero, zero

        oracle_residual = (y - out["base"].detach()).detach()
        eps = float(getattr(self.args, "residual_direction_eps", 0.02))
        direction_target = torch.ones_like(oracle_residual, dtype=torch.long)
        direction_target = torch.where(oracle_residual > eps, torch.full_like(direction_target, 2), direction_target)
        direction_target = torch.where(oracle_residual < -eps, torch.zeros_like(direction_target), direction_target)
        mag_target = torch.log1p(oracle_residual.abs())
        elem_weight = distill_weight.detach().view(-1, 1, 1)

        def direction_ce(logits: torch.Tensor) -> torch.Tensor:
            ce = F.cross_entropy(
                logits.reshape(-1, 3),
                direction_target.reshape(-1),
                reduction="none",
            ).reshape_as(oracle_residual)
            return self._weighted_mean(ce, elem_weight)

        def magnitude_loss(pred: torch.Tensor) -> torch.Tensor:
            loss = F.smooth_l1_loss(pred, mag_target, beta=self.args.huber_beta, reduction="none")
            return self._weighted_mean(loss, elem_weight)

        student_dir = direction_ce(out["student_res_dir_logits"])
        student_mag = magnitude_loss(out["student_res_mag"])
        teacher_dir = zero
        teacher_mag = zero
        if "teacher_res_dir_logits" in out:
            teacher_dir = direction_ce(out["teacher_res_dir_logits"])
            teacher_mag = magnitude_loss(out["teacher_res_mag"])
        aux_loss = student_dir + self.args.residual_aux_teacher_weight * teacher_dir + self.args.residual_mag_weight * (
            student_mag + self.args.residual_aux_teacher_weight * teacher_mag
        )
        return aux_loss, student_dir, student_mag, teacher_dir

    def _loss(
        self,
        out: dict[str, torch.Tensor],
        noisy_out: dict[str, torch.Tensor] | None,
        y: torch.Tensor,
        aug_out: dict[str, torch.Tensor] | None = None,
        contrast_outs: list[dict[str, torch.Tensor]] | None = None,
        aug_quality: float = 1.0,
        noisy_quality: float = 0.0,
        soft_teacher: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        criterion = self._select_criterion()

        pred_loss = criterion(out["student"], y)
        base_loss = criterion(out["base"], y)
        has_teacher = self._teacher_enabled() and "teacher" in out
        with torch.no_grad():
            base_mse_det = torch.mean((out["base"] - y) ** 2, dim=(1, 2))
            if has_teacher:
                teacher_mse_det = torch.mean((out["teacher"] - y) ** 2, dim=(1, 2))
                teacher_gain_det = base_mse_det - teacher_mse_det
                teacher_advantage = torch.sigmoid(teacher_gain_det / self.args.distill_temperature)
                distill_weight = teacher_advantage
                if bool(getattr(self.args, "use_selective_distill", 0)):
                    threshold = float(getattr(self.args, "selective_gain_threshold", 0.0))
                    distill_weight = torch.sigmoid((teacher_gain_det - threshold) / self.args.distill_temperature)
                    min_weight = float(getattr(self.args, "selective_min_weight", 0.0))
                    if min_weight > 0:
                        distill_weight = min_weight + (1.0 - min_weight) * distill_weight
                if self.args.teacher_advantage_floor > 0:
                    teacher_advantage = torch.clamp(teacher_advantage, min=self.args.teacher_advantage_floor)
                    distill_weight = torch.clamp(distill_weight, min=self.args.teacher_advantage_floor)
                reliability_advantage = distill_weight if bool(getattr(self.args, "use_counterfactual_reliability", 0)) else teacher_advantage
            else:
                teacher_advantage = torch.ones_like(base_mse_det)
                distill_weight = torch.ones_like(base_mse_det)
                reliability_advantage = teacher_advantage
                teacher_gain_det = torch.zeros_like(base_mse_det)

        zero = out["student"].new_tensor(0.0)
        if has_teacher:
            teacher_loss = criterion(out["teacher"], y)
            pred_distill = torch.mean(torch.mean((out["student"] - out["teacher"].detach()) ** 2, dim=(1, 2)) * distill_weight)
            plan_distill = torch.mean(torch.mean((out["pred_res_sem"] - out["true_res_sem"].detach()) ** 2, dim=-1) * distill_weight)
            token_distill = zero
            if "student_plan_token" in out and "teacher_plan_token" in out and out["student_plan_token"].shape == out["teacher_plan_token"].shape:
                token_distill = torch.mean(
                    torch.mean((out["student_plan_token"] - out["teacher_plan_token"].detach()) ** 2, dim=-1) * distill_weight
                )
            student_residual = out["student_gate"] * out["student_delta"]
            teacher_residual = out["teacher_gate"].detach() * out["teacher_delta"].detach()
            residual_distill = torch.mean(torch.mean((student_residual - teacher_residual) ** 2, dim=(1, 2)) * distill_weight)
            pattern_distill, pattern_scale, pattern_period = self._privileged_residual_pattern_loss(
                student_residual,
                teacher_residual,
                distill_weight,
            )
            coeff_distill = zero
            if "student_coeff" in out and "teacher_coeff" in out:
                coeff_distill = F.kl_div(
                    torch.log(out["student_coeff"].clamp_min(1e-6)),
                    out["teacher_coeff"].detach().clamp_min(1e-6),
                    reduction="batchmean",
                )
            gate_target = reliability_advantage.view(-1, 1, 1) * out["teacher_gate"].detach()
            gate_mse = F.mse_loss(out["student_gate"], gate_target.detach())
            gate_lazy = torch.relu(gate_target.detach() - out["student_gate"]).pow(2).mean()
            gate_guidance = gate_mse + self.args.gate_lazy_weight * gate_lazy
            residual_fit = F.mse_loss(out["student"], y)
            distill_loss = (
                self.args.distill_pred_weight * pred_distill
                + self.args.distill_plan_weight * (plan_distill + 0.2 * token_distill)
                + self.args.distill_residual_weight * residual_distill
                + self.args.distill_pattern_weight * pattern_distill
                + self.args.distill_coeff_weight * coeff_distill
                + self.args.distill_gate_weight * gate_guidance
                + self.args.distill_fit_weight * residual_fit
            )
            teacher_mse = torch.mean((out["teacher"] - y) ** 2, dim=(1, 2))
            teacher_margin = torch.relu(teacher_mse - base_mse_det + self.args.teacher_margin).mean()
            residual_aux_loss, residual_dir_loss, residual_mag_loss, teacher_dir_loss = self._residual_aux_loss(
                out, y, distill_weight
            )
        else:
            teacher_loss = zero
            distill_loss = zero
            residual_distill = zero
            pattern_distill = zero
            pattern_scale = zero
            pattern_period = zero
            token_distill = zero
            coeff_distill = zero
            gate_guidance = zero
            residual_fit = zero
            teacher_margin = zero
            residual_aux_loss = zero
            residual_dir_loss = zero
            residual_mag_loss = zero
            teacher_dir_loss = zero

        view_pred_loss = zero
        if aug_out is not None and float(getattr(self.args, "lambda_view_pred", 0.0)) > 0:
            view_pred_loss = criterion(aug_out["student"], y)
            view_consistency_weight = float(getattr(self.args, "view_consistency_weight", 0.2))
            if view_consistency_weight > 0:
                view_pred_loss = view_pred_loss + view_consistency_weight * F.mse_loss(
                    aug_out["student"], out["student"].detach()
                )

        quality_loss = zero
        if self._quality_enabled() and noisy_out is not None:
            clean_target = torch.ones_like(out["student_reliability"])
            reliability_target = getattr(self.args, "reliability_target", "quality")
            if has_teacher and reliability_target in {"teacher_advantage", "quality_advantage"}:
                clean_target = reliability_advantage.detach().view(-1, 1).to(clean_target.dtype)
            noisy_target = torch.full_like(noisy_out["student_reliability"], noisy_quality)
            if has_teacher and reliability_target == "quality_advantage":
                noisy_target = noisy_target * clean_target
            reliability_loss = F.binary_cross_entropy(out["student_reliability"], clean_target) + F.binary_cross_entropy(
                noisy_out["student_reliability"], noisy_target
            )
            consistency_loss = zero
            if aug_out is not None:
                aug_target = torch.full_like(aug_out["student_reliability"], aug_quality)
                if has_teacher and reliability_target == "quality_advantage":
                    aug_target = aug_target * clean_target
                elif has_teacher and reliability_target == "teacher_advantage":
                    aug_target = clean_target
                reliability_loss = reliability_loss + 0.5 * F.binary_cross_entropy(aug_out["student_reliability"], aug_target)
                consistency_loss = F.mse_loss(aug_out["student"], out["student"].detach()) + 0.2 * F.mse_loss(
                    aug_out["pred_res_sem"], out["pred_res_sem"].detach()
                )
            noisy_fallback = F.mse_loss(noisy_out["student"], out["base"].detach()) + 0.1 * noisy_out["student_gate"].mean()
            reliability_order = torch.relu(
                noisy_out["student_reliability"] - out["student_reliability"].detach() + self.args.reliability_margin
            ).mean()
            gate_order = torch.relu(
                noisy_out["student_gate"].mean(dim=(1, 2), keepdim=True)
                - out["student_gate"].detach().mean(dim=(1, 2), keepdim=True)
                + self.args.gate_margin
            ).mean()
            quality_loss = reliability_loss + noisy_fallback + consistency_loss + 0.5 * (reliability_order + gate_order)

        text_contrast_loss = zero
        if noisy_out is not None and float(getattr(self.args, "lambda_text_contrast", 0.0)) > 0:
            bad_outs = [noisy_out]
            if contrast_outs:
                bad_outs.extend(contrast_outs)
            clean_mse_for_text = torch.mean((out["student"] - y) ** 2, dim=(1, 2))
            margin = float(getattr(self.args, "text_contrast_margin", 0.0))
            rank_terms = []
            for bad_out in bad_outs:
                bad_mse_for_text = torch.mean((bad_out["student"] - y) ** 2, dim=(1, 2))
                normalized_gap = (
                    torch.relu(clean_mse_for_text - bad_mse_for_text + margin)
                    / base_mse_det.detach().clamp_min(1e-6)
                )
                rank_terms.append(normalized_gap.mean())
                if float(getattr(self.args, "text_contrast_rank_weight", 0.0)) > 0:
                    bad_rel = bad_out["student_reliability"]
                    clean_rel = out["student_reliability"].detach()
                    rel_rank = torch.relu(bad_rel - clean_rel + self.args.reliability_margin).mean()
                    bad_gate = bad_out["student_gate"].mean(dim=(1, 2), keepdim=True)
                    clean_gate = out["student_gate"].detach().mean(dim=(1, 2), keepdim=True)
                    gate_rank = torch.relu(bad_gate - clean_gate + self.args.gate_margin).mean()
                    rank_terms.append(float(getattr(self.args, "text_contrast_rank_weight", 0.0)) * (rel_rank + 0.5 * gate_rank))
                if float(getattr(self.args, "text_contrast_fallback_weight", 0.0)) > 0:
                    bad_fallback = F.mse_loss(bad_out["student"], out["base"].detach()) + 0.1 * bad_out["student_gate"].mean()
                    rank_terms.append(float(getattr(self.args, "text_contrast_fallback_weight", 0.0)) * bad_fallback)
            text_contrast_loss = torch.stack(rank_terms).mean() if rank_terms else zero

        student_mse = torch.mean((out["student"] - y) ** 2, dim=(1, 2))
        oracle_residual_loss = zero
        if float(getattr(self.args, "lambda_oracle_residual", 0.0)) > 0:
            oracle_residual = (y - out["base"].detach()).detach()
            student_residual_out = out["student"] - out["base"].detach()
            oracle_residual_loss = F.mse_loss(student_residual_out, oracle_residual)

        advantage_transfer_loss = zero
        if has_teacher and float(getattr(self.args, "lambda_advantage_transfer", 0.0)) > 0:
            student_gain = base_mse_det.detach() - student_mse
            target_advantage = teacher_gain_det.detach().clamp_min(0.0) * float(
                getattr(self.args, "advantage_transfer_fraction", 0.5)
            )
            margin = float(getattr(self.args, "advantage_transfer_margin", 0.0))
            relative_gap = torch.relu(target_advantage - student_gain + margin) / base_mse_det.detach().clamp_min(1e-6)
            advantage_transfer_loss = self._sample_weighted_mean(relative_gap, distill_weight.detach())

        student_margin = torch.relu(student_mse - base_mse_det + self.args.student_margin).mean()
        safety_loss = student_margin + 0.2 * teacher_margin
        phys_loss = out.get("numeric_phys_loss", zero)
        soft_teacher_loss = F.mse_loss(out["student"], soft_teacher) if soft_teacher is not None else zero

        lambda_safety = self.args.lambda_safety
        if bool(getattr(self, "_reliability_warmup_active", False)) and bool(
            getattr(self.args, "disable_safety_during_reliability_warmup", 1)
        ):
            lambda_safety = 0.0

        adaptive_logs: dict[str, float] = {}
        if bool(getattr(self, "_teacher_warmup_active", False)) and has_teacher:
            loss = self.args.lambda_teacher * teacher_loss + self.args.lambda_base * base_loss + self.args.lambda_phys * phys_loss
        elif self.loss_balancer is not None:
            adaptive_aux, adaptive_logs = self.loss_balancer(
                {
                    "teacher": teacher_loss,
                    "distill": distill_loss,
                    "soft_teacher": soft_teacher_loss,
                    "quality": quality_loss,
                    "safety": safety_loss,
                },
                {
                    "teacher": self.args.lambda_teacher,
                    "distill": self.args.lambda_distill,
                    "soft_teacher": self.args.lambda_soft_teacher,
                    "quality": self.args.lambda_quality,
                    "safety": lambda_safety,
                },
            )
            loss = (
                pred_loss
                + self.args.lambda_base * base_loss
                + self.args.lambda_residual_aux * residual_aux_loss
                + self.args.lambda_view_pred * view_pred_loss
                + self.args.lambda_oracle_residual * oracle_residual_loss
                + self.args.lambda_advantage_transfer * advantage_transfer_loss
                + self.args.lambda_phys * phys_loss
                + adaptive_aux
            )
        else:
            loss = (
                pred_loss
                + self.args.lambda_teacher * teacher_loss
                + self.args.lambda_base * base_loss
                + self.args.lambda_distill * distill_loss
                + self.args.lambda_residual_aux * residual_aux_loss
                + self.args.lambda_soft_teacher * soft_teacher_loss
                + self.args.lambda_view_pred * view_pred_loss
                + self.args.lambda_oracle_residual * oracle_residual_loss
                + self.args.lambda_advantage_transfer * advantage_transfer_loss
                + self.args.lambda_quality * quality_loss
                + self.args.lambda_text_contrast * text_contrast_loss
                + lambda_safety * safety_loss
                + self.args.lambda_phys * phys_loss
            )
        logs = {
            "loss": float(loss.item()),
            "pred": float(pred_loss.item()),
            "teacher": float(teacher_loss.item()),
            "base": float(base_loss.item()),
            "distill": float(distill_loss.item()),
            "plan_distill": float(plan_distill.item()) if has_teacher else 0.0,
            "token_distill": float(token_distill.item()),
            "coeff_distill": float(coeff_distill.item()),
            "pred_distill": float(pred_distill.item()) if has_teacher else 0.0,
            "residual_distill": float(residual_distill.item()),
            "pattern_distill": float(pattern_distill.item()),
            "pattern_scale": float(pattern_scale.item()),
            "pattern_period": float(pattern_period.item()),
            "residual_fit": float(residual_fit.item()),
            "residual_aux": float(residual_aux_loss.item()),
            "residual_dir": float(residual_dir_loss.item()),
            "residual_mag": float(residual_mag_loss.item()),
            "teacher_dir": float(teacher_dir_loss.item()),
            "gate_guidance": float(gate_guidance.item()),
            "view_pred": float(view_pred_loss.item()),
            "oracle_residual": float(oracle_residual_loss.item()),
            "advantage_transfer": float(advantage_transfer_loss.item()),
            "quality": float(quality_loss.item()),
            "text_contrast": float(text_contrast_loss.item()),
            "safety": float(safety_loss.item()),
            "lambda_safety_eff": float(lambda_safety),
            "soft_teacher": float(soft_teacher_loss.item()),
            "teacher_advantage": float(teacher_advantage.mean().item()),
            "teacher_gain": float(teacher_gain_det.mean().item()),
            "distill_weight": float(distill_weight.mean().item()),
            "selective_rate": float((distill_weight > 0.5).float().mean().item()),
            "phys": float(phys_loss.item()),
        }
        logs.update(adaptive_logs)
        return loss, logs

    @torch.no_grad()
    def vali(self, vali_data, vali_loader, criterion) -> float:
        self.model.eval()
        losses: list[float] = []
        for batch_x, batch_y, history_text, teacher_text, _noise_text, _row_idx in vali_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            history_text = history_text.float().to(self.device)
            teacher_text = teacher_text.float().to(self.device)
            out = self.model(batch_x, history_text, teacher_text if self._teacher_enabled() else None)
            val_metric = getattr(self.args, "val_metric", "loss")
            if val_metric == "mse":
                value = F.mse_loss(out["student"], batch_y)
            elif val_metric == "mae":
                value = F.l1_loss(out["student"], batch_y)
            else:
                value = criterion(out["student"], batch_y)
            losses.append(float(value.item()))
        self.model.train()
        return float(np.average(losses))

    def train(self, setting: str):


        train_data, train_loader = self._get_data("train")
        vali_data, vali_loader = self._get_data("val")
        test_data, test_loader = self._get_data("test")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        time_now = time.time()
        train_steps = len(train_loader)
        self._load_pretrained_numeric_checkpoint()
        self._freeze_numeric_backbone()
        soft_teacher_array = self._load_soft_teacher()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        history: list[dict[str, float | int]] = []

        print(f"Train/Val/Test windows: {len(train_data)}/{len(vali_data)}/{len(test_data)}")
        print(f"Split info: {train_data.split_info}")
        for epoch in range(self.args.train_epochs):
            warmup_active = epoch < int(getattr(self.args, "reliability_warmup_epochs", 0))
            teacher_warmup_active = self._teacher_enabled() and epoch < int(
                getattr(self.args, "semantic_teacher_warmup_epochs", 0)
            )
            self._reliability_warmup_active = warmup_active
            self._teacher_warmup_active = teacher_warmup_active
            self._set_reliability_warmup(warmup_active)
            if warmup_active:
                print(f"Reliability warm-up active for epoch {epoch + 1}: student gate bypasses reliability.")
            if teacher_warmup_active:
                print(f"Privileged teacher warm-up active for epoch {epoch + 1}: loss updates teacher path only.")
            iter_count = 0
            train_logs: dict[str, float] = {}
            self.model.train()
            self._keep_frozen_numeric_eval()
            epoch_time = time.time()
            positive_cols = self._arg_columns(
                self.args.positive_text_cols, [self.args.student_text_col, "paraphrase_text", "compact_text"]
            )
            negative_cols = self._arg_columns(self.args.negative_text_cols, UNRELIABLE_COLUMNS)
            for batch_x, batch_y, history_text, teacher_text, _noise_text, row_idx in train_loader:
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                history_text = history_text.float().to(self.device)
                teacher_text = teacher_text.float().to(self.device)
                soft_teacher = None
                if soft_teacher_array is not None and self.args.lambda_soft_teacher > 0:
                    soft_teacher = torch.from_numpy(soft_teacher_array[row_idx.detach().cpu().numpy()]).float().to(self.device)
                aug_out = None
                noisy_out = None
                aug_quality = 1.0
                noisy_quality = 0.0

                out = self.model(batch_x, history_text, teacher_text if self._teacher_enabled() else None)
                if self._quality_enabled() and not teacher_warmup_active:
                    aug_text, _aug_col, aug_quality = self._sample_text_batch(train_data, row_idx, positive_cols, self.device)
                    noise_text, _noise_col, noisy_quality = self._sample_text_batch(train_data, row_idx, negative_cols, self.device)
                    aug_out = self.model(batch_x, history_text, None, override_history_text=aug_text)
                    noisy_out = self.model(batch_x, history_text, None, override_history_text=noise_text)
                contrast_outs = None
                if (
                    bool(getattr(self.args, "use_explicit_text_contrast", 0))
                    and float(getattr(self.args, "lambda_text_contrast", 0.0)) > 0
                    and not teacher_warmup_active
                ):
                    contrast_outs = []
                    if bool(getattr(self.args, "text_contrast_include_shuffle", 1)) and history_text.shape[0] > 1:
                        perm = torch.randperm(history_text.shape[0], device=history_text.device)
                        if torch.all(perm == torch.arange(history_text.shape[0], device=history_text.device)):
                            perm = torch.roll(perm, shifts=1)
                        shuffled_text = history_text[perm]
                        contrast_outs.append(self.model(batch_x, history_text, None, override_history_text=shuffled_text))
                    if bool(getattr(self.args, "text_contrast_include_no_text", 1)) and "missing_text" in train_data.text_features:
                        missing_text = self._student_override_batch(train_data, row_idx, "missing_text", self.device)
                        contrast_outs.append(self.model(batch_x, history_text, None, override_history_text=missing_text))
                    if noisy_out is None and contrast_outs:
                        noisy_out = contrast_outs[0]
                loss, logs = self._loss(
                    out,
                    noisy_out,
                    batch_y,
                    aug_out=aug_out,
                    contrast_outs=contrast_outs,
                    aug_quality=aug_quality,
                    noisy_quality=noisy_quality,
                    soft_teacher=soft_teacher,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                model_optim.step()

                bsz = batch_x.shape[0]
                for key, value in logs.items():
                    train_logs[key] = train_logs.get(key, 0.0) + value * bsz

            train_logs = {key: value / len(train_data) for key, value in train_logs.items()}
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, Time: {time.time() - epoch_time:.2f}s | "
                f"Train Loss: {train_logs['loss']:.6f} Vali Loss: {vali_loss:.6f} Test Loss: {test_loss:.6f}"
            )
            history.append({"epoch": epoch + 1, "vali_loss": vali_loss, "test_loss": test_loss, **train_logs})
            if warmup_active or teacher_warmup_active:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
                continue
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        try:
            state_dict = torch.load(best_model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self._reliability_warmup_active = False
        self._teacher_warmup_active = False
        self._set_reliability_warmup(False)
        history_path = Path(path) / "train_history.json"
        # Some Windows runs produce long checkpoint paths; use the extended
        # prefix for this auxiliary file so training does not fail after saving
        # a valid model checkpoint.
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path_to_write = history_path.resolve()
        if os.name == "nt":
            history_path_str = str(history_path_to_write)
            if not history_path_str.startswith("\\\\?\\"):
                history_path_to_write = Path("\\\\?\\" + history_path_str)
        history_path_to_write.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Training finished in {time.time() - time_now:.2f}s")
        self._print_numeric_backbone_info("Training finished")
        return self.model

    @torch.no_grad()
    def _evaluate_setting(
        self,
        test_data,
        test_loader,
        noise_col: str | None = None,
        residual_calibration: dict[str, float] | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        preds, trues, bases, teachers, gates, reliabilities = [], [], [], [], [], []
        aux_stats = {
            "student_dir_correct": 0.0,
            "teacher_dir_correct": 0.0,
            "dir_count": 0.0,
            "student_mag_abs": 0.0,
            "teacher_mag_abs": 0.0,
            "mag_count": 0.0,
        }
        for batch_x, batch_y, history_text, teacher_text, _noise_text, row_idx in test_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            history_text = history_text.float().to(self.device)
            teacher_text = teacher_text.float().to(self.device)
            if noise_col is None:
                out = self.model(batch_x, history_text, teacher_text if self._teacher_enabled() else None)
                out = self._apply_prompt_ensemble(out, batch_x, history_text, test_data, row_idx)
            elif noise_col == "__shuffled__":
                shift = max(1, history_text.shape[0] // 2)
                override = torch.roll(history_text, shifts=shift, dims=0) if history_text.shape[0] > 1 else history_text
                out = self.model(
                    batch_x,
                    history_text,
                    teacher_text if self._teacher_enabled() else None,
                    override_history_text=override,
                )
            else:
                override = self._student_override_batch(test_data, row_idx, noise_col, self.device)
                out = self.model(
                    batch_x,
                    history_text,
                    teacher_text if self._teacher_enabled() else None,
                    override_history_text=override,
                )
            preds.append(out["student"].detach().cpu().numpy())
            bases.append(out["base"].detach().cpu().numpy())
            teachers.append(out.get("teacher", out["base"]).detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            gates.append(out["student_gate"].detach().cpu().numpy())
            reliabilities.append(out["student_reliability"].detach().cpu().numpy())
            if "student_res_dir_logits" in out:
                oracle_residual = batch_y - out["base"]
                eps = float(getattr(self.args, "residual_direction_eps", 0.02))
                direction_target = torch.ones_like(oracle_residual, dtype=torch.long)
                direction_target = torch.where(oracle_residual > eps, torch.full_like(direction_target, 2), direction_target)
                direction_target = torch.where(oracle_residual < -eps, torch.zeros_like(direction_target), direction_target)
                target_count = float(direction_target.numel())
                aux_stats["dir_count"] += target_count
                student_dir = out["student_res_dir_logits"].argmax(dim=-1)
                aux_stats["student_dir_correct"] += float((student_dir == direction_target).float().sum().item())
                if "teacher_res_dir_logits" in out:
                    teacher_dir = out["teacher_res_dir_logits"].argmax(dim=-1)
                    aux_stats["teacher_dir_correct"] += float((teacher_dir == direction_target).float().sum().item())
                mag_target = torch.log1p(oracle_residual.abs())
                aux_stats["mag_count"] += target_count
                aux_stats["student_mag_abs"] += float((out["student_res_mag"] - mag_target).abs().sum().item())
                if "teacher_res_mag" in out:
                    aux_stats["teacher_mag_abs"] += float((out["teacher_res_mag"] - mag_target).abs().sum().item())

        pred = np.concatenate(preds, axis=0)
        true = np.concatenate(trues, axis=0)
        base = np.concatenate(bases, axis=0)
        teacher = np.concatenate(teachers, axis=0)
        gate = np.concatenate(gates, axis=0)
        reliability = np.concatenate(reliabilities, axis=0)
        pred = self._apply_residual_calibration(pred, base, gate, reliability, residual_calibration)

        pred_inv = test_data.inverse_transform(pred)
        true_inv = test_data.inverse_transform(true)
        base_inv = test_data.inverse_transform(base)
        teacher_inv = test_data.inverse_transform(teacher)

        pred_eval, true_eval, base_eval, teacher_eval = (pred_inv, true_inv, base_inv, teacher_inv) if self.args.inverse else (pred, true, base, teacher)

        mae, mse, rmse, mape, mspe = metric(pred_eval, true_eval)
        base_mae, base_mse, base_rmse, base_mape, base_mspe = metric(base_eval, true_eval)
        teacher_mae, teacher_mse, teacher_rmse, teacher_mape, teacher_mspe = metric(teacher_eval, true_eval)
        inv_mae, inv_mse, inv_rmse, inv_mape, inv_mspe = metric(pred_inv, true_inv)
        inv_base_mae, inv_base_mse, inv_base_rmse, inv_base_mape, inv_base_mspe = metric(base_inv, true_inv)
        inv_teacher_mae, inv_teacher_mse, inv_teacher_rmse, inv_teacher_mape, inv_teacher_mspe = metric(teacher_inv, true_inv)
        out = {
            "student_mae": mae,
            "student_mse": mse,
            "student_rmse": rmse,
            "student_mape": mape,
            "student_mspe": mspe,
            "numeric_base_mae": base_mae,
            "numeric_base_mse": base_mse,
            "numeric_base_rmse": base_rmse,
            "numeric_base_mape": base_mape,
            "numeric_base_mspe": base_mspe,
            "teacher_oracle_mae": teacher_mae,
            "teacher_oracle_mse": teacher_mse,
            "teacher_oracle_rmse": teacher_rmse,
            "teacher_oracle_mape": teacher_mape,
            "teacher_oracle_mspe": teacher_mspe,
            "student_ntr": self._negative_transfer_rate(pred_eval, base_eval, true_eval, eps=self.args.ntr_eps),
            "teacher_oracle_ntr": self._negative_transfer_rate(teacher_eval, base_eval, true_eval, eps=self.args.ntr_eps),
            "ntr_eps": float(self.args.ntr_eps),
            "mean_gate": float(gate.mean()),
            "mean_reliability": float(reliability.mean()),
        }
        if residual_calibration:
            out.update(
                {
                    "residual_calibration_scale": float(residual_calibration.get("scale", 1.0)),
                    "residual_calibration_threshold": float(residual_calibration.get("threshold", 0.0)),
                    "residual_calibration_val_mse": float(residual_calibration.get("val_mse", float("nan"))),
                    "residual_calibration_val_base_mse": float(residual_calibration.get("val_base_mse", float("nan"))),
                    "residual_calibration_score": str(residual_calibration.get("score_mode", "rel_gate")),
                }
            )
        out.update(self._transfer_diagnostics(pred_eval, base_eval, true_eval, gate, reliability, eps=self.args.ntr_eps))
        if aux_stats["dir_count"] > 0:
            out.update(
                {
                    "student_residual_dir_acc": aux_stats["student_dir_correct"] / aux_stats["dir_count"],
                    "teacher_residual_dir_acc": aux_stats["teacher_dir_correct"] / aux_stats["dir_count"],
                    "student_residual_mag_mae": aux_stats["student_mag_abs"] / aux_stats["mag_count"],
                    "teacher_residual_mag_mae": aux_stats["teacher_mag_abs"] / aux_stats["mag_count"],
                }
            )
        if not self.args.inverse:
            out.update(
                {
                    "student_mae_inverse": inv_mae,
                    "student_mse_inverse": inv_mse,
                    "student_rmse_inverse": inv_rmse,
                    "student_mape_inverse": inv_mape,
                    "student_mspe_inverse": inv_mspe,
                    "numeric_base_mae_inverse": inv_base_mae,
                    "numeric_base_mse_inverse": inv_base_mse,
                    "numeric_base_rmse_inverse": inv_base_rmse,
                    "numeric_base_mape_inverse": inv_base_mape,
                    "numeric_base_mspe_inverse": inv_base_mspe,
                    "teacher_oracle_mae_inverse": inv_teacher_mae,
                    "teacher_oracle_mse_inverse": inv_teacher_mse,
                    "teacher_oracle_rmse_inverse": inv_teacher_rmse,
                    "teacher_oracle_mape_inverse": inv_teacher_mape,
                    "teacher_oracle_mspe_inverse": inv_teacher_mspe,
                }
            )
        return out

    @torch.no_grad()
    def _deploy_equivalence_check(self, test_loader) -> dict[str, float]:
        """Verify that test-time student predictions do not depend on teacher text."""
        self.model.eval()
        max_diffs = {
            "student_max_abs_diff": 0.0,
            "base_max_abs_diff": 0.0,
            "gate_max_abs_diff": 0.0,
            "reliability_max_abs_diff": 0.0,
        }
        if not self._teacher_enabled():
            return max_diffs
        for batch_x, _batch_y, history_text, teacher_text, _noise_text, _row_idx in test_loader:
            batch_x = batch_x.float().to(self.device)
            history_text = history_text.float().to(self.device)
            teacher_text = teacher_text.float().to(self.device)
            with_teacher = self.model(batch_x, history_text, teacher_text)
            deploy_only = self.model(batch_x, history_text, None)
            pairs = {
                "student_max_abs_diff": ("student", "student"),
                "base_max_abs_diff": ("base", "base"),
                "gate_max_abs_diff": ("student_gate", "student_gate"),
                "reliability_max_abs_diff": ("student_reliability", "student_reliability"),
            }
            for out_key, (left_key, right_key) in pairs.items():
                diff = (with_teacher[left_key] - deploy_only[right_key]).abs().max().item()
                max_diffs[out_key] = max(max_diffs[out_key], float(diff))
        return max_diffs

    def test(self, setting: str, test: int = 0):
        test_data, test_loader = self._get_data("test")
        if test:
            checkpoint = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            try:
                state_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state_dict)
        residual_calibration = None
        if bool(getattr(self.args, "calibrate_residual", 0)):
            vali_data, vali_loader = self._get_data("val")
            residual_calibration = self._calibrate_residual_on_validation(vali_data, vali_loader)
            print(f"Validation residual calibration: {json.dumps(residual_calibration, indent=2)}")

        folder_path = Path(self.args.output_dir) / setting
        folder_path.mkdir(parents=True, exist_ok=True)
        results: dict[str, object] = {
            "args": vars(self.args),
            "split_info": test_data.split_info,
            "metric_scale": "inverse original units" if self.args.inverse else "standardized values from data loader; *_inverse keys are original units",
            "residual_calibration": residual_calibration,
            "test_clean": self._evaluate_setting(test_data, test_loader, residual_calibration=residual_calibration),
            "deploy_equivalence_check": self._deploy_equivalence_check(test_loader),
        }
        numeric_info = self._numeric_backbone_info()
        if numeric_info is not None:
            results["numeric_backbone_info"] = numeric_info
        for col in UNRELIABLE_COLUMNS + ["paraphrase_text", "compact_text"]:
            results[f"test_{col}"] = self._evaluate_setting(
                test_data,
                test_loader,
                noise_col=col,
                residual_calibration=residual_calibration,
            )
        if bool(getattr(self.args, "evaluate_shuffled_text", 1)):
            results["test_shuffled_text"] = self._evaluate_setting(
                test_data,
                test_loader,
                noise_col="__shuffled__",
                residual_calibration=residual_calibration,
            )

        metrics_path = folder_path / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        clean_metrics = results["test_clean"]
        results_line = (
            f"{setting}\n"
            f"mse:{float(clean_metrics['student_mse'])}, mae:{float(clean_metrics['student_mae'])}\n\n"
        )
        with Path("results.txt").open("a", encoding="utf-8") as f:
            f.write(results_line)
        print(f"Wrote metrics to {metrics_path.resolve()}")
        print(f"Appended final MSE/MAE to {(Path.cwd() / 'results.txt').resolve()}")
        self._print_numeric_backbone_info("Testing")
        print(json.dumps(results["test_clean"], indent=2))
        return results["test_clean"]

