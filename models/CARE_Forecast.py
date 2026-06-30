from __future__ import annotations

import torch
import torch.fft as fft
from torch import nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalResidualPlanner(nn.Module):
    """Horizon-token residual planner with semantic cross-attention.

    The numeric backbone stays text-free and produces a physical base forecast.
    Text conditions only the residual planning tokens, which is aligned with the
    paper story: semantic context explains how to correct the numeric dynamics.
    """

    def __init__(
        self,
        hidden_dim: int,
        sem_dim: int,
        pred_len: int,
        c_out: int,
        dropout: float,
        n_heads: int = 4,
        use_pseudo_sem: bool = False,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.c_out = c_out
        self.use_pseudo_sem = use_pseudo_sem
        self.horizon_tokens = nn.Parameter(torch.randn(1, pred_len, hidden_dim) * 0.02)
        self.base_query_proj = nn.Sequential(nn.Linear(c_out, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.hidden_query_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.numeric_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.hist_proj = nn.Sequential(nn.Linear(sem_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.plan_proj = nn.Sequential(nn.Linear(sem_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.pseudo_proj = nn.Sequential(nn.Linear(sem_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.query_film = nn.Sequential(nn.Linear(sem_dim * (3 if use_pseudo_sem else 2), hidden_dim * 2), nn.Tanh())
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.delta_head = nn.Linear(hidden_dim, c_out)
        self.gate_head = nn.Linear(hidden_dim, c_out)
        self.delta_scale_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.delta_head.bias)
        nn.init.constant_(self.gate_head.bias, 0.5)

    def forward(
        self,
        hidden: torch.Tensor,
        hist_sem: torch.Tensor,
        plan_sem: torch.Tensor,
        pseudo_sem: torch.Tensor | None = None,
        base_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = hidden.shape[0]
        context_tokens = [
            self.numeric_proj(hidden).unsqueeze(1),
            self.hist_proj(hist_sem).unsqueeze(1),
            self.plan_proj(plan_sem).unsqueeze(1),
        ]
        film_inputs = [hist_sem, plan_sem]
        if self.use_pseudo_sem and pseudo_sem is not None:
            context_tokens.append(self.pseudo_proj(pseudo_sem).unsqueeze(1))
            film_inputs.append(pseudo_sem)

        context = torch.cat(context_tokens, dim=1)
        gamma_beta = self.query_film(torch.cat(film_inputs, dim=-1)).chunk(2, dim=-1)
        gamma, beta = gamma_beta[0].unsqueeze(1), gamma_beta[1].unsqueeze(1)
        query = self.horizon_tokens.expand(bsz, -1, -1) + self.hidden_query_proj(hidden).unsqueeze(1)
        if base_norm is not None:
            query = query + self.base_query_proj(base_norm)
        query = query * (1.0 + 0.1 * gamma) + 0.1 * beta
        query = self.query_norm(query)
        attended, _ = self.cross_attn(query, context, context, need_weights=False)
        tokens = query + attended
        tokens = tokens + self.ffn(tokens)
        delta_scale = torch.sigmoid(self.delta_scale_logit)
        return delta_scale * self.delta_head(tokens), self.gate_head(tokens), tokens.mean(dim=1)


class ModReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(x)
        scale = torch.relu(mag + self.bias) / (mag + 1e-6)
        return x * scale


class ComplexTemporalMixer(nn.Module):
    """Complex MLP along the temporal/fractional-frequency axis.

    The mixer is shared by all variables and never treats variables as tokens,
    so it keeps the numeric branch lighter than the iTransformer-style module
    in the original PhyFSME implementation.
    """

    def __init__(self, seq_len: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w_re = nn.Linear(seq_len, hidden_dim)
        self.w_im = nn.Linear(seq_len, hidden_dim)
        self.v_re = nn.Linear(hidden_dim, seq_len)
        self.v_im = nn.Linear(hidden_dim, seq_len)
        self.act = ModReLU()
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _complex_linear(x: torch.Tensor, w_re: nn.Linear, w_im: nn.Linear) -> torch.Tensor:
        real = w_re(x.real) - w_im(x.imag)
        imag = w_re(x.imag) + w_im(x.real)
        return torch.complex(real, imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self._complex_linear(x, self.w_re, self.w_im)
        out = self.act(out)
        out = torch.complex(self.dropout(out.real), self.dropout(out.imag))
        out = self._complex_linear(out, self.v_re, self.v_im)
        return residual + out


class ExponentialMovingAverage(nn.Module):
    """xPatch-style EMA trend extraction, implemented device-safely."""

    def __init__(self, alpha: float):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32).clamp(1e-4, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = x.new_empty(x.shape)
        state = x[:, 0, :]
        trend[:, 0, :] = state
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        for step in range(1, x.shape[1]):
            state = alpha * x[:, step, :] + (1.0 - alpha) * state
            trend[:, step, :] = state
        return trend


class DoubleExponentialMovingAverage(nn.Module):
    """Optional DEMA branch from xPatch for stronger trend smoothing."""

    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32).clamp(1e-4, 1.0))
        self.register_buffer("beta", torch.tensor(float(beta), dtype=torch.float32).clamp(1e-4, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = x.new_empty(x.shape)
        state = x[:, 0, :]
        slope = x[:, 1, :] - state if x.shape[1] > 1 else torch.zeros_like(state)
        trend[:, 0, :] = state
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)
        beta = self.beta.to(dtype=x.dtype, device=x.device)
        for step in range(1, x.shape[1]):
            prev_state = state
            state = alpha * x[:, step, :] + (1.0 - alpha) * (state + slope)
            slope = beta * (state - prev_state) + (1.0 - beta) * slope
            trend[:, step, :] = state
        return trend


class MovingAverage(nn.Module):
    """Parallel moving-average trend extraction used in common TSF decompositions."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"moving_avg_kernel must be positive, got {kernel_size}")
        self.kernel_size = int(kernel_size)
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]. Replicate-padding avoids endpoint shrinkage.
        x_t = x.transpose(1, 2)
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x_pad = F.pad(x_t, (pad_left, pad_right), mode="replicate")
        return self.pool(x_pad).transpose(1, 2)


class XPatchDecomposition(nn.Module):
    """Decompose a normalized series into seasonal/residual and trend parts."""

    def __init__(
        self,
        decomp_type: str = "moving_avg",
        alpha: float = 0.3,
        beta: float = 0.3,
        moving_avg_kernel: int = 25,
    ):
        super().__init__()
        self.decomp_type = decomp_type
        if decomp_type == "moving_avg":
            self.ma = MovingAverage(moving_avg_kernel)
        elif decomp_type == "ema":
            self.ma = ExponentialMovingAverage(alpha)
        elif decomp_type == "dema":
            self.ma = DoubleExponentialMovingAverage(alpha, beta)
        else:
            raise ValueError(f"Unsupported decomp_type: {decomp_type}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.ma(x)
        seasonal = x - trend
        return seasonal, trend


class FixedFRFTTransform(nn.Module):
    """Fixed-order FRFT transform shared by the trend branch."""

    def __init__(
        self,
        patch_len: int,
        init_alpha: float = 0.4,
    ):
        super().__init__()
        self.patch_len = patch_len
        init_alpha_tensor = torch.tensor(float(init_alpha), dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
        self.register_buffer("fixed_alpha", init_alpha_tensor.view(1))

    def current_alpha(self) -> torch.Tensor:
        return self.fixed_alpha

    def compute_physical_loss(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        imag = x.imag
        imag_smooth = torch.abs(imag[:, :, 1:] - imag[:, :, :-1]).mean()
        imag_abs = torch.abs(imag).mean()
        imag_energy = imag.pow(2).mean()
        real_energy = x.real.pow(2).mean()
        ratio = imag_energy / (real_energy + imag_energy + 1e-8)
        ratio_loss = torch.relu(ratio - alpha.detach())
        return ratio_loss * 0.01 + imag_smooth * 0.1 + imag_abs * 0.1

    def frft_3d(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        bsz, channels, seq_len = x.shape
        device = x.device
        alpha = alpha % 4
        if alpha > 2:
            alpha = alpha - 4
        if alpha < -2:
            alpha = alpha + 4

        if torch.abs(alpha) < 1e-4:
            return x.to(torch.complex64)
        if torch.abs(alpha - 1.0) < 1e-4:
            return fft.fftshift(fft.fft(fft.ifftshift(x, dim=-1), dim=-1, norm="ortho"), dim=-1)
        if torch.abs(alpha + 1.0) < 1e-4:
            return fft.fftshift(fft.ifft(fft.ifftshift(x, dim=-1), dim=-1, norm="ortho"), dim=-1)

        x_flat = x.to(torch.complex64).reshape(-1, seq_len)
        theta = alpha * torch.pi / 2
        n = torch.arange(seq_len, device=device).float() - seq_len // 2
        t_sq = n.pow(2) / seq_len
        c1 = torch.exp(-1j * torch.pi * torch.tan(theta / 2) * t_sq)
        c2 = torch.exp(-1j * torch.pi * torch.sin(theta) * t_sq)
        a_alpha = torch.exp(-1j * (torch.pi * torch.tanh(100 * torch.sin(theta)) / 4 - theta / 2))

        res = x_flat * c1
        res = fft.ifftshift(res, dim=-1)
        res = fft.fft(res, dim=-1, norm="ortho")
        res = fft.fftshift(res, dim=-1)
        res = res * c2
        res = fft.ifftshift(res, dim=-1)
        res = fft.ifft(res, dim=-1, norm="ortho")
        res = fft.fftshift(res, dim=-1)
        res = res * c1 * a_alpha
        return res.reshape(bsz, channels, seq_len)


class SeasonalFFTBlock(nn.Module):
    """Periodic branch: standard Fourier transform, complex MLP, inverse FFT."""

    def __init__(self, seq_len: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.seq_len = seq_len
        freq_len = seq_len // 2 + 1
        self.mixer_1 = ComplexTemporalMixer(freq_len, hidden_dim, dropout)
        self.mixer_2 = ComplexTemporalMixer(freq_len, hidden_dim, dropout)

    def forward(self, seasonal: torch.Tensor) -> torch.Tensor:
        x_freq = fft.rfft(seasonal, dim=-1, norm="ortho")
        x_freq = self.mixer_1(x_freq)
        x_freq = self.mixer_2(x_freq)
        return fft.irfft(x_freq, n=self.seq_len, dim=-1, norm="ortho")


class TrendFRFTBlock(nn.Module):
    """Trend branch: fixed-order FRFT, complex MLP, inverse FRFT."""

    def __init__(self, patch_len: int, hidden_dim: int, dropout: float, init_alpha: float = 0.4):
        super().__init__()
        self.transform = FixedFRFTTransform(patch_len=patch_len, init_alpha=init_alpha)
        self.mixer_1 = ComplexTemporalMixer(patch_len, hidden_dim, dropout)
        self.mixer_2 = ComplexTemporalMixer(patch_len, hidden_dim, dropout)

    @property
    def patch_len(self) -> int:
        return self.transform.patch_len

    def current_alpha(self) -> torch.Tensor:
        return self.transform.current_alpha()

    def forward(self, trend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, channels, seq_len = trend.shape
        if seq_len % self.patch_len != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by frft_patch_len={self.patch_len}")

        num_patches = seq_len // self.patch_len
        x_patch = trend.reshape(bsz, channels, num_patches, self.patch_len)
        x_patch = x_patch.permute(0, 2, 1, 3).contiguous().reshape(bsz * num_patches, channels, self.patch_len)

        alpha = self.current_alpha()
        x_freq = self.transform.frft_3d(x_patch, alpha)
        x_freq = self.mixer_1(x_freq)
        x_freq = self.mixer_2(x_freq)
        x_time_complex = self.transform.frft_3d(x_freq, -alpha)
        phys_loss = self.transform.compute_physical_loss(x_time_complex, alpha)

        x_time = x_time_complex.real.reshape(bsz, num_patches, channels, self.patch_len)
        x_time = x_time.permute(0, 2, 1, 3).contiguous().reshape(bsz, channels, seq_len)
        return x_time, phys_loss, alpha


class SingleScaleFRFTBackbone(nn.Module):
    """Moving-average decomposition + FFT seasonal stream + fixed-order FRFT trend stream.

    This is intentionally a compact numeric backbone: no PatchTST branch, no
    global FRFT residual branch, and no variable-token attention. The goal is to
    keep the paper's numeric contribution focused on spectral physical dynamics.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int,
        target_idx: int,
        hidden_dim: int,
        dropout: float,
        patch_len: int,
        init_alpha: float,
        decomp_type: str = "moving_avg",
        ema_alpha: float = 0.3,
        dema_beta: float = 0.3,
        moving_avg_kernel: int = 25,
        spectral_mix_init: float = -1.5,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.c_out = c_out
        self.enc_in = enc_in
        self.target_idx = target_idx
        self.decomp = XPatchDecomposition(
            decomp_type=decomp_type,
            alpha=ema_alpha,
            beta=dema_beta,
            moving_avg_kernel=moving_avg_kernel,
        )
        self.seasonal_fft = SeasonalFFTBlock(seq_len=seq_len, hidden_dim=hidden_dim, dropout=dropout)
        self.trend_frft = TrendFRFTBlock(
            patch_len=patch_len,
            hidden_dim=hidden_dim,
            dropout=dropout,
            init_alpha=init_alpha,
        )
        self.seasonal_norm = nn.LayerNorm(seq_len)
        self.trend_norm = nn.LayerNorm(seq_len)
        self.spectral_mix_logit = nn.Parameter(torch.tensor(float(spectral_mix_init)))
        self.channel_summary = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.hidden_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.linear_head = nn.Linear(seq_len, pred_len)
        self.seasonal_head = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        self.trend_head = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(self, centered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seasonal, trend = self.decomp(centered)
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        raw = seasonal + trend

        seasonal_time = self.seasonal_norm(self.seasonal_fft(seasonal))
        trend_time, phys_loss, alpha = self.trend_frft(trend)
        trend_time = self.trend_norm(trend_time)
        spectral_mix = torch.sigmoid(self.spectral_mix_logit)
        fused_time = raw + spectral_mix * (seasonal_time + trend_time)

        linear_delta = self.linear_head(raw).transpose(1, 2)
        hidden = self.hidden_fusion(self.channel_summary(fused_time).mean(dim=1))
        seasonal_delta = self.seasonal_head(seasonal_time).transpose(1, 2)
        trend_delta = self.trend_head(trend_time).transpose(1, 2)
        delta = 0.2 * linear_delta + spectral_mix * (seasonal_delta + trend_delta)
        if self.c_out != self.enc_in:
            delta = delta[:, :, self.target_idx : self.target_idx + 1]
        return hidden, delta, phys_loss, alpha


class Model(nn.Module):
    """CARE-Forecast in a standard long-term forecasting model interface.

    Inputs and outputs follow the common [B, L, C] convention. The data loader
    performs dataset-level standardization; this model optionally applies RevIN
    at the instance level and denormalizes its forecast back to loader scale.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.target_idx = getattr(configs, "target_idx", self.enc_in - 1)
        self.hidden_dim = configs.hidden_dim
        self.sem_dim = configs.sem_dim
        self.student_text_dim = int(getattr(configs, "student_text_dim", configs.text_dim))
        self.teacher_text_dim = int(getattr(configs, "teacher_text_dim", configs.text_dim))
        self.output_dim = self.pred_len * self.c_out
        self.use_revin = bool(getattr(configs, "use_revin", 1))
        self.numeric_backbone = getattr(configs, "numeric_backbone", "mlp")
        self.base_type = getattr(configs, "base_type", "hybrid")
        self.adapter_type = getattr(configs, "adapter_type", "basis")
        self.residual_planner_type = getattr(configs, "residual_planner_type", "mlp")
        self.basis_rank = int(getattr(configs, "basis_rank", 8))
        self.ablation = getattr(configs, "ablation", "full")
        self.numeric_only = self.ablation == "numeric_only"
        self.no_gate = self.ablation == "no_gate"
        self.direct_fusion = self.ablation == "direct_fusion"
        self.share_residual_decoder = bool(getattr(configs, "share_residual_decoder", 0))
        self.share_gate_decoder = bool(getattr(configs, "share_gate_decoder", 0))
        self.use_pseudo_future_sem = bool(getattr(configs, "use_pseudo_future_sem", 0))
        self.target_only_numeric = bool(getattr(configs, "target_only_numeric", 0)) and self.c_out == 1
        self.reliability_floor = float(getattr(configs, "reliability_floor", 0.0))
        self.residual_budget_type = getattr(configs, "residual_budget_type", "none")
        self.residual_budget_scale = float(getattr(configs, "residual_budget_scale", 1.0))
        self.residual_budget_min = float(getattr(configs, "residual_budget_min", 0.05))
        self.residual_budget_apply = getattr(configs, "residual_budget_apply", "student")
        self.student_residual_scale = float(getattr(configs, "student_residual_scale", 1.0))
        self.use_residual_aux = bool(getattr(configs, "use_residual_aux", 0))
        self.use_base_context = bool(getattr(configs, "use_base_context", 0))
        self.use_text_modulation = bool(getattr(configs, "use_text_modulation", 0))
        self.text_mod_mode = getattr(configs, "text_mod_mode", "film")
        self.text_mod_scale = float(getattr(configs, "text_mod_scale", 0.2))
        self.student_numeric_context_scale = float(getattr(configs, "student_numeric_context_scale", 1.0))
        self.student_base_context_scale = float(getattr(configs, "student_base_context_scale", 1.0))

        numeric_input_dim = 1 if self.target_only_numeric else self.enc_in
        numeric_dim = self.seq_len * numeric_input_dim
        self.revin = RevIN(self.enc_in, affine=True) if self.use_revin else None
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        if self.numeric_backbone == "frft":
            output_scale = self.pred_len if self.seq_len % self.pred_len == 0 else self.seq_len
            frft_patch_len = int(getattr(configs, "frft_patch_len", 0) or output_scale)
            if self.seq_len % frft_patch_len != 0:
                raise ValueError(f"frft_patch_len={frft_patch_len} must divide seq_len={self.seq_len}")
            self.frft_backbone = SingleScaleFRFTBackbone(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                enc_in=numeric_input_dim,
                c_out=self.c_out,
                target_idx=0 if self.target_only_numeric else self.target_idx,
                hidden_dim=self.hidden_dim,
                dropout=configs.dropout,
                patch_len=frft_patch_len,
                init_alpha=float(getattr(configs, "frft_init_alpha", 0.4)),
                decomp_type=getattr(configs, "decomp_type", "moving_avg"),
                ema_alpha=float(getattr(configs, "ema_alpha", 0.3)),
                dema_beta=float(getattr(configs, "dema_beta", 0.3)),
                moving_avg_kernel=int(getattr(configs, "moving_avg_kernel", 7)),
                spectral_mix_init=float(getattr(configs, "spectral_mix_init", -1.5)),
            )
        else:
            self.frft_backbone = None
        self.base_head = nn.Linear(self.hidden_dim, self.output_dim)
        self.channel_linear = nn.Linear(self.seq_len, self.pred_len)
        self.base_mix_logit = nn.Parameter(torch.tensor(-1.0))
        self.frft_mix_logit = nn.Parameter(torch.tensor(0.0))

        self.text_adapter = nn.Sequential(
            nn.Linear(self.student_text_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.hidden_dim, self.sem_dim),
            nn.LayerNorm(self.sem_dim),
        )
        self.teacher_text_adapter = None
        if self.teacher_text_dim != self.student_text_dim:
            self.teacher_text_adapter = nn.Sequential(
                nn.Linear(self.teacher_text_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.hidden_dim, self.sem_dim),
                nn.LayerNorm(self.sem_dim),
            )
        if self.use_pseudo_future_sem:
            self.pseudo_future_adapter = nn.Sequential(
                nn.Linear(self.c_out * 12, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.hidden_dim, self.sem_dim),
                nn.LayerNorm(self.sem_dim),
            )
            pseudo_sem_dim = self.sem_dim
        else:
            self.pseudo_future_adapter = None
            pseudo_sem_dim = 0
        if self.use_base_context:
            self.base_context_adapter = nn.Sequential(
                nn.Linear(self.c_out * 8, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.hidden_dim, self.sem_dim),
                nn.LayerNorm(self.sem_dim),
            )
            base_context_dim = self.sem_dim
        else:
            self.base_context_adapter = None
            base_context_dim = 0
        self.numeric_sem_proj = nn.Linear(self.hidden_dim, self.sem_dim)
        self.text_sem_proj = nn.Linear(self.sem_dim, self.sem_dim)
        self.planner = MLP(
            self.hidden_dim + self.sem_dim + pseudo_sem_dim + base_context_dim,
            self.hidden_dim,
            self.sem_dim,
            configs.dropout,
        )
        self.student_reliability = nn.Sequential(
            nn.Linear(self.hidden_dim + self.sem_dim + pseudo_sem_dim + base_context_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        fusion_dim = self.hidden_dim + self.sem_dim * 2 + pseudo_sem_dim + base_context_dim
        self.student_cross_planner = None
        self.teacher_cross_planner = None
        if self.residual_planner_type == "cross_attn":
            if self.adapter_type != "direct":
                raise ValueError("residual_planner_type=cross_attn requires adapter_type=direct.")
            planner_heads = int(getattr(configs, "planner_heads", 4))
            self.student_cross_planner = CrossModalResidualPlanner(
                self.hidden_dim,
                self.sem_dim,
                self.pred_len,
                self.c_out,
                configs.dropout,
                n_heads=planner_heads,
                use_pseudo_sem=self.use_pseudo_future_sem,
            )
            self.teacher_cross_planner = CrossModalResidualPlanner(
                self.hidden_dim,
                self.sem_dim,
                self.pred_len,
                self.c_out,
                configs.dropout,
                n_heads=planner_heads,
                use_pseudo_sem=self.use_pseudo_future_sem,
            )
        if self.adapter_type == "basis":
            self.residual_basis = MLP(self.hidden_dim, self.hidden_dim, self.output_dim * self.basis_rank, configs.dropout)
            self.teacher_coeff = MLP(fusion_dim, self.hidden_dim, self.basis_rank, configs.dropout)
            self.student_coeff = self.teacher_coeff if self.share_residual_decoder else MLP(
                fusion_dim, self.hidden_dim, self.basis_rank, configs.dropout
            )
        elif self.adapter_type == "direct":
            self.teacher_delta = MLP(fusion_dim, self.hidden_dim, self.output_dim, configs.dropout)
            self.student_delta = self.teacher_delta if self.share_residual_decoder else MLP(
                fusion_dim, self.hidden_dim, self.output_dim, configs.dropout
            )
        else:
            raise ValueError(f"Unsupported adapter_type: {self.adapter_type}")
        if self.residual_planner_type not in {"mlp", "cross_attn"}:
            raise ValueError(f"Unsupported residual_planner_type: {self.residual_planner_type}")
        if self.numeric_backbone not in {"mlp", "frft"}:
            raise ValueError(f"Unsupported numeric_backbone: {self.numeric_backbone}")
        if self.base_type not in {"mlp", "nlinear", "hybrid", "frft", "frft_hybrid"}:
            raise ValueError(f"Unsupported base_type: {self.base_type}")
        self.teacher_gate = MLP(fusion_dim, self.hidden_dim, self.output_dim, configs.dropout)
        self.student_gate = self.teacher_gate if self.share_gate_decoder else MLP(fusion_dim, self.hidden_dim, self.output_dim, configs.dropout)
        self.direct_student_head = MLP(fusion_dim, self.hidden_dim, self.output_dim, configs.dropout)
        self.direct_teacher_head = MLP(fusion_dim, self.hidden_dim, self.output_dim, configs.dropout)
        if self.use_text_modulation:
            self.student_text_modulator = MLP(self.sem_dim, self.hidden_dim, self.output_dim * 2, configs.dropout)
            if bool(getattr(configs, "text_mod_zero_init", 1)):
                nn.init.zeros_(self.student_text_modulator.net[-1].weight)
                nn.init.zeros_(self.student_text_modulator.net[-1].bias)
        else:
            self.student_text_modulator = None
        if self.use_residual_aux:
            self.student_res_dir_head = MLP(fusion_dim, self.hidden_dim, self.output_dim * 3, configs.dropout)
            self.teacher_res_dir_head = MLP(fusion_dim, self.hidden_dim, self.output_dim * 3, configs.dropout)
            self.student_res_mag_head = MLP(fusion_dim, self.hidden_dim, self.output_dim, configs.dropout)
            self.teacher_res_mag_head = MLP(fusion_dim, self.hidden_dim, self.output_dim, configs.dropout)
        else:
            self.student_res_dir_head = None
            self.teacher_res_dir_head = None
            self.student_res_mag_head = None
            self.teacher_res_mag_head = None
        nn.init.constant_(self.student_gate.net[-1].bias, 0.5)
        nn.init.constant_(self.teacher_gate.net[-1].bias, 0.5)
        nn.init.constant_(self.student_reliability[-1].bias, 0.5)

    def _basis_delta(self, hidden: torch.Tensor, adapter_input: torch.Tensor, teacher: bool) -> tuple[torch.Tensor, torch.Tensor]:
        basis = self.residual_basis(hidden).reshape(-1, self.basis_rank, self.pred_len, self.c_out)
        coeff_head = self.teacher_coeff if teacher else self.student_coeff
        coeff = torch.softmax(coeff_head(adapter_input), dim=-1)
        return torch.einsum("bk,bkhc->bhc", coeff, basis), coeff

    def _residual_delta(
        self,
        hidden: torch.Tensor,
        adapter_input: torch.Tensor,
        teacher: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.adapter_type == "basis":
            return self._basis_delta(hidden, adapter_input, teacher=teacher)
        head = self.teacher_delta if teacher else self.student_delta
        return head(adapter_input).reshape(-1, self.pred_len, self.c_out), None

    def _residual_aux(self, adapter_input: torch.Tensor, teacher: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.use_residual_aux:
            return None
        dir_head = self.teacher_res_dir_head if teacher else self.student_res_dir_head
        mag_head = self.teacher_res_mag_head if teacher else self.student_res_mag_head
        if dir_head is None or mag_head is None:
            return None
        direction = dir_head(adapter_input).reshape(-1, self.pred_len, self.c_out, 3)
        magnitude = F.softplus(mag_head(adapter_input).reshape(-1, self.pred_len, self.c_out))
        return direction, magnitude

    def _last_level(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.c_out == self.enc_in:
            return x_norm[:, -1:, :].repeat(1, self.pred_len, 1)
        return x_norm[:, -1:, self.target_idx : self.target_idx + 1].repeat(1, self.pred_len, 1)

    def _output_history(self, x_norm: torch.Tensor) -> torch.Tensor:
        if self.c_out == self.enc_in:
            return x_norm
        return x_norm[:, :, self.target_idx : self.target_idx + 1]

    def _residual_budget(self, x_norm: torch.Tensor) -> torch.Tensor | None:
        if self.residual_budget_type == "none":
            return None
        history = self._output_history(x_norm)
        recent = history[:, -min(history.shape[1], self.pred_len) :, :]
        if self.residual_budget_type == "history_std":
            scale = recent.std(dim=1, keepdim=True, unbiased=False)
        elif self.residual_budget_type == "history_mad":
            center = recent.median(dim=1, keepdim=True).values
            scale = (recent - center).abs().median(dim=1, keepdim=True).values * 1.4826
        else:
            raise ValueError(f"Unsupported residual_budget_type: {self.residual_budget_type}")
        return self.residual_budget_scale * scale.clamp_min(self.residual_budget_min)

    @staticmethod
    def _apply_residual_budget(delta: torch.Tensor, budget: torch.Tensor | None) -> torch.Tensor:
        if budget is None:
            return delta
        return budget * torch.tanh(delta / (budget + 1e-6))

    def _apply_text_modulation(
        self,
        delta: torch.Tensor,
        hist_sem: torch.Tensor,
        budget: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.student_text_modulator is None or self.text_mod_scale <= 0:
            return delta
        gamma_beta = self.student_text_modulator(hist_sem).reshape(-1, self.pred_len, self.c_out, 2)
        gamma, beta = gamma_beta[..., 0], gamma_beta[..., 1]
        scale = self.text_mod_scale
        if budget is None:
            beta_term = torch.tanh(beta)
        else:
            beta_term = budget * torch.tanh(beta)
        if self.text_mod_mode in {"gated", "forced_gated"}:
            # A gated mode makes the residual path explicitly pass through text
            # features instead of behaving as an almost pure numeric adapter.
            text_conditioned = torch.sigmoid(gamma) * delta + beta_term
            return (1.0 - scale) * delta + scale * text_conditioned
        if self.text_mod_mode != "film":
            raise ValueError(f"Unsupported text_mod_mode: {self.text_mod_mode}")
        return delta * (1.0 + scale * torch.tanh(gamma)) + scale * beta_term

    def _pseudo_future_semantic(self, x_norm: torch.Tensor, base_norm: torch.Tensor) -> torch.Tensor | None:
        if self.pseudo_future_adapter is None:
            return None
        history = self._output_history(x_norm)
        recent = history[:, -min(history.shape[1], self.pred_len) :, :]
        last = history[:, -1:, :]
        delta = base_norm - last
        base_diff = base_norm[:, 1:, :] - base_norm[:, :-1, :]
        recent_diff = recent[:, 1:, :] - recent[:, :-1, :] if recent.shape[1] > 1 else torch.zeros_like(recent)
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
        return self.pseudo_future_adapter(torch.cat(stats, dim=-1))

    def _base_context_semantic(self, base_norm: torch.Tensor) -> torch.Tensor | None:
        if self.base_context_adapter is None:
            return None
        diff = base_norm[:, 1:, :] - base_norm[:, :-1, :] if base_norm.shape[1] > 1 else torch.zeros_like(base_norm)
        stats = [
            base_norm.mean(dim=1),
            base_norm.std(dim=1, unbiased=False),
            base_norm[:, 0, :],
            base_norm[:, -1, :],
            base_norm.amin(dim=1),
            base_norm.amax(dim=1),
            diff.mean(dim=1),
            diff.std(dim=1, unbiased=False),
        ]
        return self.base_context_adapter(torch.cat(stats, dim=-1))

    def _denorm_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        if self.revin is None:
            return y_norm
        if self.c_out == self.enc_in:
            return self.revin(y_norm, "denorm")
        if self.revin._mean is None or self.revin._stdev is None:
            raise RuntimeError("RevIN state is missing for target-channel denorm.")
        out = y_norm
        if self.revin.affine:
            weight = self.revin.weight[:, :, self.target_idx : self.target_idx + 1]
            bias = self.revin.bias[:, :, self.target_idx : self.target_idx + 1]
            out = (out - bias) / (weight + self.revin.eps * self.revin.eps)
        mean = self.revin._mean[:, :, self.target_idx : self.target_idx + 1]
        stdev = self.revin._stdev[:, :, self.target_idx : self.target_idx + 1]
        return out * stdev + mean

    def _encode_numeric(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x_norm = self.revin(x_enc, "norm") if self.revin is not None else x_enc
        centered = x_norm - x_norm[:, -1:, :]
        numeric_centered = self._output_history(centered) if self.target_only_numeric else centered
        aux: dict[str, torch.Tensor] = {}
        if self.numeric_backbone == "frft":
            if self.frft_backbone is None:
                raise RuntimeError("FRFT backbone is not initialized.")
            hidden, frft_delta, phys_loss, alpha = self.frft_backbone(numeric_centered)
            aux.update({"frft_delta": frft_delta, "numeric_phys_loss": phys_loss, "frft_alpha": alpha})
        else:
            hidden = self.numeric_encoder(numeric_centered.reshape(numeric_centered.shape[0], -1))
            aux.update(
                {
                    "numeric_phys_loss": x_enc.new_tensor(0.0),
                    "frft_alpha": x_enc.new_tensor(0.0),
                }
            )
        return hidden, x_norm, centered, aux

    def _base_forecast(
        self,
        hidden: torch.Tensor,
        centered: torch.Tensor,
        x_norm: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        mlp_delta = self.base_head(hidden).reshape(-1, self.pred_len, self.c_out)
        linear_delta = self.channel_linear(centered.transpose(1, 2)).transpose(1, 2)
        if self.c_out != self.enc_in:
            linear_delta = linear_delta[:, :, self.target_idx : self.target_idx + 1]
        frft_delta = aux.get("frft_delta", mlp_delta)

        if self.base_type == "mlp":
            base_delta = mlp_delta
        elif self.base_type == "nlinear":
            base_delta = linear_delta
        elif self.base_type == "frft":
            base_delta = frft_delta
        elif self.base_type == "frft_hybrid":
            mix = torch.sigmoid(self.frft_mix_logit)
            base_delta = mix * frft_delta + (1.0 - mix) * linear_delta
        else:
            mix = torch.sigmoid(self.base_mix_logit)
            base_delta = mix * mlp_delta + (1.0 - mix) * linear_delta
        return base_delta + self._last_level(x_norm)

    def _cross_modal_consistency(self, hidden: torch.Tensor, sem: torch.Tensor) -> torch.Tensor:
        numeric_sem = F.normalize(self.numeric_sem_proj(hidden), dim=-1)
        text_sem = F.normalize(self.text_sem_proj(sem), dim=-1)
        return F.cosine_similarity(numeric_sem, text_sem, dim=-1, eps=1e-6).unsqueeze(-1)

    def _encode_teacher_text(self, teacher_text: torch.Tensor) -> torch.Tensor:
        expected_dim = self.teacher_text_dim
        if teacher_text.shape[-1] != expected_dim:
            raise ValueError(f"teacher_text dim={teacher_text.shape[-1]} does not match teacher_text_dim={expected_dim}")
        if self.teacher_text_adapter is None:
            return self.text_adapter(teacher_text)
        return self.teacher_text_adapter(teacher_text)

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
        pseudo_sem = self._pseudo_future_semantic(x_norm, base_norm)
        base_context_sem = self._base_context_semantic(base_norm)
        student_hidden = hidden * self.student_numeric_context_scale
        student_base_context_sem = (
            base_context_sem * self.student_base_context_scale if base_context_sem is not None else None
        )
        reliability_inputs = [student_hidden, hist_sem]
        planner_inputs = [student_hidden, hist_sem]
        if pseudo_sem is not None:
            reliability_inputs.append(pseudo_sem)
            planner_inputs.append(pseudo_sem)
        if student_base_context_sem is not None:
            reliability_inputs.append(student_base_context_sem)
            planner_inputs.append(student_base_context_sem)
        reliability_inputs.append(consistency)
        reliability = torch.sigmoid(self.student_reliability(torch.cat(reliability_inputs, dim=-1)))
        gate_reliability = reliability
        if bool(getattr(self, "runtime_reliability_bypass", False)):
            gate_reliability = torch.ones_like(reliability)
        if self.reliability_floor > 0:
            gate_reliability = self.reliability_floor + (1.0 - self.reliability_floor) * reliability
        pred_res_sem = self.planner(torch.cat(planner_inputs, dim=-1))
        student_parts = [student_hidden, hist_sem, pred_res_sem]
        if pseudo_sem is not None:
            student_parts.append(pseudo_sem)
        if student_base_context_sem is not None:
            student_parts.append(student_base_context_sem)
        student_in = torch.cat(student_parts, dim=-1)
        if self.residual_planner_type == "cross_attn":
            if self.student_cross_planner is None:
                raise RuntimeError("student_cross_planner is not initialized.")
            student_delta, student_gate_logits, student_token = self.student_cross_planner(
                student_hidden, hist_sem, pred_res_sem, pseudo_sem, base_norm=base_norm
            )
            raw_student_gate = torch.sigmoid(student_gate_logits)
            student_coeff = None
        else:
            raw_student_gate = torch.sigmoid(self.student_gate(student_in).reshape(-1, self.pred_len, self.c_out))
            student_delta, student_coeff = self._residual_delta(student_hidden, student_in, teacher=False)
            student_token = pred_res_sem
        student_aux = self._residual_aux(student_in, teacher=False)
        student_delta = self._apply_text_modulation(student_delta, hist_sem, residual_budget)
        student_delta = self._apply_residual_budget(student_delta, residual_budget)
        student_gate = torch.ones_like(raw_student_gate) if self.no_gate else gate_reliability.unsqueeze(-1) * raw_student_gate
        if self.direct_fusion:
            student_delta = self.direct_student_head(student_in).reshape(-1, self.pred_len, self.c_out)
            student_norm = self._last_level(x_norm) + student_delta
        else:
            student_norm = base_norm + self.student_residual_scale * student_gate * student_delta

        out = {
            "base": self._denorm_output(base_norm),
            "student": self._denorm_output(student_norm),
            "student_delta": student_delta,
            "student_gate": student_gate,
            "student_reliability": reliability,
            "pred_res_sem": pred_res_sem,
            "student_plan_token": student_token,
            "hist_sem": hist_sem,
            "base_context_sem": base_context_sem if base_context_sem is not None else torch.zeros_like(hist_sem),
            "text_numeric_consistency": consistency,
            "numeric_phys_loss": numeric_aux["numeric_phys_loss"],
            "frft_alpha": numeric_aux["frft_alpha"],
        }
        if student_coeff is not None:
            out["student_coeff"] = student_coeff
        if student_aux is not None:
            out["student_res_dir_logits"], out["student_res_mag"] = student_aux

        if teacher_text is not None:
            privileged_sem = self._encode_teacher_text(teacher_text)
            teacher_parts = [hidden, hist_sem, privileged_sem]
            if pseudo_sem is not None:
                teacher_parts.append(pseudo_sem)
            if base_context_sem is not None:
                teacher_parts.append(base_context_sem)
            teacher_in = torch.cat(teacher_parts, dim=-1)
            if self.residual_planner_type == "cross_attn":
                if self.teacher_cross_planner is None:
                    raise RuntimeError("teacher_cross_planner is not initialized.")
                teacher_delta, teacher_gate_logits, teacher_token = self.teacher_cross_planner(
                    hidden, hist_sem, privileged_sem, pseudo_sem, base_norm=base_norm
                )
                raw_teacher_gate = torch.sigmoid(teacher_gate_logits)
                teacher_coeff = None
            else:
                raw_teacher_gate = torch.sigmoid(self.teacher_gate(teacher_in).reshape(-1, self.pred_len, self.c_out))
                teacher_delta, teacher_coeff = self._residual_delta(hidden, teacher_in, teacher=True)
                teacher_token = privileged_sem
            teacher_aux = self._residual_aux(teacher_in, teacher=True)
            if self.residual_budget_apply == "both":
                teacher_delta = self._apply_residual_budget(teacher_delta, residual_budget)
            teacher_gate = torch.ones_like(raw_teacher_gate) if self.no_gate else raw_teacher_gate
            if self.direct_fusion:
                teacher_delta = self.direct_teacher_head(teacher_in).reshape(-1, self.pred_len, self.c_out)
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
                    "teacher_plan_token": teacher_token,
                }
            )
            if teacher_coeff is not None:
                out["teacher_coeff"] = teacher_coeff
            if teacher_aux is not None:
                out["teacher_res_dir_logits"], out["teacher_res_mag"] = teacher_aux
        return out
