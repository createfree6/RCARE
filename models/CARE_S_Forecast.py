from __future__ import annotations

import torch
from torch import nn

from .CARE_Forecast import Model as CAREForecastModel


class Model(CAREForecastModel):
    """Single-variable CARE variant with a privileged-semantic bridge.

    This model keeps the MA-FRFT numeric backbone from CARE_Forecast, but replaces
    the student/teacher residual component with a shared horizon decoder:

    - teacher path: historical text + privileged future/residual text;
    - student path: historical text + predicted privileged semantic code;
    - shared decoder: same residual/gate manifold for both paths.
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.single_gate_floor = float(getattr(configs, "single_gate_floor", 0.0))
        self.single_gate_floor = max(0.0, min(0.8, self.single_gate_floor))
        self.single_teacher_gate_floor = float(getattr(configs, "single_teacher_gate_floor", 0.0))
        self.single_teacher_gate_floor = max(0.0, min(0.8, self.single_teacher_gate_floor))
        self.stat_dim = self.c_out * 12
        self.numeric_future_adapter = nn.Sequential(
            nn.Linear(self.stat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.hidden_dim, self.sem_dim),
            nn.LayerNorm(self.sem_dim),
        )
        self.privileged_bridge = nn.Sequential(
            nn.Linear(self.hidden_dim + self.sem_dim * 2 + 1, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.hidden_dim, self.sem_dim),
        )
        self.bridge_norm = nn.LayerNorm(self.sem_dim)
        self.bridge_confidence = nn.Sequential(
            nn.Linear(self.hidden_dim + self.sem_dim * 2 + 1, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        nn.init.constant_(self.bridge_confidence[-1].bias, 0.5)
        self.shared_head_mix_logit = nn.Parameter(torch.tensor(-1.5))

    def _future_stats(self, x_norm: torch.Tensor, base_norm: torch.Tensor) -> torch.Tensor:
        history = self._output_history(x_norm)
        recent = history[:, -min(history.shape[1], self.pred_len) :, :]
        last = history[:, -1:, :]
        base_diff = base_norm[:, 1:, :] - base_norm[:, :-1, :]
        recent_diff = recent[:, 1:, :] - recent[:, :-1, :] if recent.shape[1] > 1 else torch.zeros_like(recent)
        delta = base_norm - last
        stats = [
            base_norm.mean(dim=1),
            base_norm.std(dim=1, unbiased=False),
            base_norm[:, 0, :],
            base_norm[:, -1, :],
            base_norm.amin(dim=1),
            base_norm.amax(dim=1),
            delta.mean(dim=1),
            delta[:, -1, :],
            base_diff.mean(dim=1),
            base_diff.std(dim=1, unbiased=False),
            recent.mean(dim=1),
            recent_diff.mean(dim=1),
        ]
        return torch.cat(stats, dim=-1)

    @staticmethod
    def _gate_with_floor(gate: torch.Tensor, floor: float) -> torch.Tensor:
        if floor <= 0:
            return gate
        return floor + (1.0 - floor) * gate

    def forward(
        self,
        x_enc: torch.Tensor,
        history_text: torch.Tensor,
        teacher_text: torch.Tensor | None = None,
        override_history_text: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden, x_norm, centered, numeric_aux = self._encode_numeric(x_enc)
        base_norm = self._base_forecast(hidden, centered, x_norm, numeric_aux)
        residual_budget = self._residual_budget(x_norm)
        if self.numeric_only:
            zeros_delta = torch.zeros_like(base_norm)
            zeros_sem = torch.zeros(x_enc.shape[0], self.sem_dim, device=x_enc.device, dtype=x_enc.dtype)
            zeros_rel = torch.zeros(x_enc.shape[0], 1, device=x_enc.device, dtype=x_enc.dtype)
            out = {
                "base": self._denorm_output(base_norm),
                "student": self._denorm_output(base_norm),
                "student_delta": zeros_delta,
                "student_gate": zeros_delta,
                "student_reliability": zeros_rel,
                "pred_res_sem": zeros_sem,
                "hist_sem": zeros_sem,
                "text_numeric_consistency": zeros_rel,
                "numeric_phys_loss": numeric_aux["numeric_phys_loss"],
                "frft_alpha": numeric_aux["frft_alpha"],
            }
            if teacher_text is not None:
                out.update(
                    {
                        "teacher": self._denorm_output(base_norm),
                        "teacher_delta": zeros_delta,
                        "teacher_gate": zeros_delta,
                        "true_res_sem": zeros_sem,
                    }
                )
            return out

        hist_input = override_history_text if override_history_text is not None else history_text
        hist_sem = self.text_adapter(hist_input)
        consistency = self._cross_modal_consistency(hidden, hist_sem)
        numeric_future_sem = self.numeric_future_adapter(self._future_stats(x_norm, base_norm))
        bridge_input = torch.cat([hidden, hist_sem, numeric_future_sem, consistency], dim=-1)
        bridge_delta = self.privileged_bridge(bridge_input)
        bridge_conf = torch.sigmoid(self.bridge_confidence(bridge_input))
        pred_priv_sem = self.bridge_norm(numeric_future_sem + bridge_conf * bridge_delta)

        reliability_inputs = [hidden, hist_sem]
        pseudo_sem = self._pseudo_future_semantic(x_norm, base_norm)
        if pseudo_sem is not None:
            reliability_inputs.append(pseudo_sem)
        reliability_inputs.append(consistency)
        reliability = torch.sigmoid(self.student_reliability(torch.cat(reliability_inputs, dim=-1)))
        gate_reliability = reliability
        if bool(getattr(self, "runtime_reliability_bypass", False)):
            gate_reliability = torch.ones_like(reliability)
        gate_reliability = self._gate_with_floor(gate_reliability, max(self.reliability_floor, self.single_gate_floor))

        student_parts = [hidden, hist_sem, pred_priv_sem]
        if pseudo_sem is not None:
            student_parts.append(pseudo_sem)
        student_in = torch.cat(student_parts, dim=-1)
        teacher_style_delta, teacher_style_coeff = self._residual_delta(hidden, student_in, teacher=True)
        student_own_delta, student_coeff = self._residual_delta(hidden, student_in, teacher=False)
        shared_mix = torch.sigmoid(self.shared_head_mix_logit)
        student_delta = shared_mix * teacher_style_delta + (1.0 - shared_mix) * student_own_delta
        student_delta = self._apply_residual_budget(student_delta, residual_budget)
        teacher_style_gate = torch.sigmoid(self.teacher_gate(student_in).reshape(-1, self.pred_len, self.c_out))
        student_own_gate = torch.sigmoid(self.student_gate(student_in).reshape(-1, self.pred_len, self.c_out))
        raw_student_gate = shared_mix * teacher_style_gate + (1.0 - shared_mix) * student_own_gate
        student_token = pred_priv_sem
        student_gate = torch.ones_like(raw_student_gate) if self.no_gate else gate_reliability.unsqueeze(-1) * raw_student_gate
        if self.direct_fusion:
            student_norm = self._last_level(x_norm) + student_delta
        else:
            student_norm = base_norm + student_gate * student_delta

        out = {
            "base": self._denorm_output(base_norm),
            "student": self._denorm_output(student_norm),
            "student_delta": student_delta,
            "student_gate": student_gate,
            "student_reliability": reliability,
            "pred_res_sem": pred_priv_sem,
            "student_plan_token": student_token,
            "hist_sem": hist_sem,
            "text_numeric_consistency": consistency,
            "numeric_phys_loss": numeric_aux["numeric_phys_loss"],
            "frft_alpha": numeric_aux["frft_alpha"],
        }
        if student_coeff is not None:
            out["student_coeff"] = student_coeff
        elif teacher_style_coeff is not None:
            out["student_coeff"] = teacher_style_coeff

        if teacher_text is not None:
            privileged_sem = self._encode_teacher_text(teacher_text)
            teacher_parts = [hidden, hist_sem, privileged_sem]
            if pseudo_sem is not None:
                teacher_parts.append(pseudo_sem)
            teacher_in = torch.cat(teacher_parts, dim=-1)
            teacher_delta, teacher_coeff = self._residual_delta(hidden, teacher_in, teacher=True)
            if self.residual_budget_apply == "both":
                teacher_delta = self._apply_residual_budget(teacher_delta, residual_budget)
            raw_teacher_gate = torch.sigmoid(self.teacher_gate(teacher_in).reshape(-1, self.pred_len, self.c_out))
            raw_teacher_gate = self._gate_with_floor(raw_teacher_gate, self.single_teacher_gate_floor)
            teacher_gate = torch.ones_like(raw_teacher_gate) if self.no_gate else raw_teacher_gate
            if self.direct_fusion:
                teacher_norm = self._last_level(x_norm) + teacher_delta
            else:
                teacher_norm = base_norm + teacher_gate * teacher_delta
            out.update(
                {
                    "teacher": self._denorm_output(teacher_norm),
                    "teacher_delta": teacher_delta,
                    "teacher_gate": teacher_gate,
                    "true_res_sem": privileged_sem,
                    "teacher_privileged_sem": privileged_sem,
                    "teacher_plan_token": privileged_sem,
                }
            )
            if teacher_coeff is not None:
                out["teacher_coeff"] = teacher_coeff
        return out
