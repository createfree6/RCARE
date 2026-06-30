from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-.]+")


@dataclass
class HashingTextVectorizer:
    """A tiny deterministic text encoder for fast local experiments.

    This is intentionally dependency-free. It lets us build and test the
    forecasting pipeline before plugging in a frozen BERT/BGE/LLM encoder.
    """

    dim: int = 256
    lowercase: bool = True
    signed: bool = True

    def encode(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            text = "No textual information available."
        if self.lowercase:
            text = text.lower()

        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = TOKEN_RE.findall(text)
        if not tokens:
            tokens = ["empty"]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="little", signed=False)
            idx = value % self.dim
            sign = -1.0 if self.signed and ((value >> 8) & 1) else 1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def encode_many(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(text) for text in texts], axis=0)


@dataclass
class HybridTextVectorizer:
    """Hash text plus explicit temporal semantics extracted from summaries.

    The hashing block keeps the pipeline model-agnostic and cheap. The small
    semantic block mimics what a frozen language encoder should expose: trend,
    volatility, uncertainty, contradiction/noise cues, and numeric summary hints.
    """

    dim: int = 256
    semantic_dim: int = 96

    def __post_init__(self) -> None:
        self.semantic_dim = min(self.semantic_dim, max(16, self.dim // 2))
        self.hash_dim = self.dim - self.semantic_dim
        self.hasher = HashingTextVectorizer(dim=max(1, self.hash_dim))

    def encode(self, text: str, summary: dict[str, Any] | None = None) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            text = "No textual information available."
        lower = text.lower()
        sem = np.zeros(self.semantic_dim, dtype=np.float32)

        self._set_phrase_features(sem, lower)
        if summary:
            self._set_summary_features(sem, summary)

        hash_vec = self.hasher.encode(text)
        out = np.concatenate([sem, hash_vec], axis=0).astype(np.float32)
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out /= norm
        return out

    def encode_many(self, texts: list[str], summaries: list[dict[str, Any] | None] | None = None) -> np.ndarray:
        if summaries is None:
            summaries = [None] * len(texts)
        return np.stack([self.encode(text, summary) for text, summary in zip(texts, summaries)], axis=0)

    @staticmethod
    def _put(vec: np.ndarray, idx: int, value: float = 1.0) -> None:
        if 0 <= idx < len(vec):
            vec[idx] = float(value)

    def _set_phrase_features(self, vec: np.ndarray, text: str) -> None:
        phrase_to_idx = {
            "upward": 0,
            "downward": 1,
            "stable": 2,
            "high volatility": 3,
            "medium volatility": 4,
            "low volatility": 5,
            "strong periodicity": 6,
            "medium periodicity": 7,
            "weak periodicity": 8,
            "recent increase": 9,
            "recent decrease": 10,
            "little recent change": 11,
            "anomaly detected: yes": 12,
            "anomaly detected: no": 13,
            "uncertainty is high": 14,
            "uncertainty is medium": 15,
            "uncertainty is low": 16,
            "positive correction": 17,
            "negative correction": 18,
            "small correction": 19,
            "teacher-only": 20,
            "contradictory context": 21,
            "irrelevant": 22,
            "no textual information": 23,
        }
        for phrase, idx in phrase_to_idx.items():
            if phrase in text:
                self._put(vec, idx)

    def _set_summary_features(self, vec: np.ndarray, summary: dict[str, Any]) -> None:
        categorical = {
            "trend": {"upward": 24, "downward": 25, "stable": 26},
            "volatility": {"high": 27, "medium": 28, "low": 29},
            "periodicity": {"strong": 30, "medium": 31, "weak": 32},
            "recent_change": {"recent increase": 33, "recent decrease": 34, "little recent change": 35},
            "anomaly": {"yes": 36, "no": 37},
            "uncertainty": {"high": 38, "medium": 39, "low": 40},
        }
        for key, mapping in categorical.items():
            value = str(summary.get(key, ""))
            idx = mapping.get(value)
            if idx is not None:
                self._put(vec, idx)

        numeric_keys = [
            "trend_score",
            "recent_change_score",
            "volatility_score",
            "period_corr",
            "anomaly_count",
            "first",
            "last",
            "mean",
            "std",
            "min",
            "max",
        ]
        start = 48
        for offset, key in enumerate(numeric_keys):
            value = safe_float(summary.get(key), 0.0)
            scale = 1.0 if key.endswith("score") or key in {"volatility_score", "period_corr", "anomaly_count"} else 10.0
            self._put(vec, start + offset, math.tanh(value / scale))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out
