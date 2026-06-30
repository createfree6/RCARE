from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalization for [B, L, C] time-series tensors."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self._mean: torch.Tensor | None = None
        self._stdev: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            var = x.var(dim=1, keepdim=True, unbiased=False)
            self._stdev = torch.sqrt(var + self.eps).detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.weight + self.bias
            return x
        if mode == "denorm":
            if self._mean is None or self._stdev is None:
                raise RuntimeError("RevIN denorm called before norm.")
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps * self.eps)
            return x * self._stdev + self._mean
        raise ValueError(f"Unsupported RevIN mode: {mode}")
