from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from care_forecast.text_features import HashingTextVectorizer, HybridTextVectorizer


TEXT_COLUMNS = [
    "history_text",
    "future_text",
    "residual_text",
    "compact_text",
    "paraphrase_text",
    "contradictory_text",
    "noisy_text",
    "missing_text",
    "time_shift_text",
    "irrelevant_text",
    "llm_history_text",
    "llm_history_future_text",
    "llm_history_prior_text",
    "llm_history_numeric_text",
    "llm_future_text",
    "llm_residual_text",
    "llm_privileged_text",
]

UNRELIABLE_COLUMNS = ["contradictory_text", "noisy_text", "missing_text", "time_shift_text", "irrelevant_text"]

SUMMARY_FOR_TEXT = {
    "history_text": "history_summary",
    "compact_text": "history_summary",
    "paraphrase_text": "history_summary",
    "future_text": "future_summary",
    "residual_text": "residual_summary",
    "llm_history_text": "history_summary",
    "llm_history_future_text": "history_future_summary",
    "llm_history_prior_text": "history_prior_summary",
    "llm_history_numeric_text": "history_numeric_summary",
    "llm_future_text": "future_summary",
    "llm_residual_text": "residual_summary",
    "llm_privileged_text": "residual_summary",
}

_TEXT_FEATURE_CACHE: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}


class StandardScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> None:
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-6

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before transform.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform.")
        return data * self.std + self.mean


def parse_summary(value: object) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_text_features(text_df: pd.DataFrame, dim: int, encoder: str, cache_key: str) -> dict[str, np.ndarray]:
    key = (cache_key, dim, encoder)
    if key in _TEXT_FEATURE_CACHE:
        return _TEXT_FEATURE_CACHE[key]

    if encoder == "hybrid":
        vectorizer = HybridTextVectorizer(dim=dim)
    elif encoder == "hash":
        vectorizer = HashingTextVectorizer(dim=dim)
    else:
        raise ValueError(f"Unsupported text encoder: {encoder}")

    features: dict[str, np.ndarray] = {}
    for col in TEXT_COLUMNS:
        if col not in text_df.columns:
            continue
        texts = text_df[col].fillna("No textual information available.").astype(str).tolist()
        summary_col = SUMMARY_FOR_TEXT.get(col)
        if encoder == "hybrid" and summary_col in text_df.columns:
            summaries = [parse_summary(value) for value in text_df[summary_col].tolist()]
            features[col] = vectorizer.encode_many(texts, summaries)
        else:
            features[col] = vectorizer.encode_many(texts)
    _TEXT_FEATURE_CACHE[key] = features
    return features


def load_text_features_from_npz(text_df: pd.DataFrame, feature_path: Path, dim: int) -> dict[str, np.ndarray]:
    arrays = np.load(feature_path)
    features: dict[str, np.ndarray] = {}
    for col in TEXT_COLUMNS:
        if col not in arrays:
            continue
        value = arrays[col].astype(np.float32)
        if value.shape[0] != len(text_df):
            raise ValueError(f"Text feature {col} has {value.shape[0]} rows, expected {len(text_df)}.")
        if value.shape[1] != dim:
            raise ValueError(f"Text feature {col} has dim={value.shape[1]}, but args.text_dim={dim}.")
        features[col] = value

    zero = np.zeros((len(text_df), dim), dtype=np.float32)
    for col in TEXT_COLUMNS:
        if col in text_df.columns and col not in features:
            features[col] = zero.copy()
    return features


def ratio_split_indices(total: int, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | str]]:
    if max_samples is not None:
        total = min(total, max_samples)
    n_train = int(total * 0.7)
    n_val = int(total * 0.1)
    train = np.arange(0, n_train)
    val = np.arange(n_train, n_train + n_val)
    test = np.arange(n_train + n_val, total)
    return train, val, test, {"mode": "ratio", "total_windows": int(total)}


def _apply_train_ratio(
    train: np.ndarray,
    ratio: float,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    if ratio <= 0 or ratio > 1:
        raise ValueError(f"train_ratio must be in (0, 1], got {ratio}")
    original_count = int(len(train))
    if ratio >= 1 or original_count <= 1:
        return train, {
            "train_ratio": float(ratio),
            "train_ratio_mode": mode,
            "original_train_windows": original_count,
            "selected_train_windows": original_count,
        }

    keep = max(1, int(round(original_count * ratio)))
    if mode == "uniform":
        positions = np.linspace(0, original_count - 1, num=keep, dtype=np.int64)
        selected = train[positions]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(original_count, size=keep, replace=False))
        selected = train[positions]
    elif mode == "prefix":
        selected = train[:keep]
    else:
        raise ValueError(f"Unsupported train_ratio_mode: {mode}")

    return selected.astype(np.int64), {
        "train_ratio": float(ratio),
        "train_ratio_mode": mode,
        "train_ratio_seed": int(seed),
        "original_train_windows": original_count,
        "selected_train_windows": int(len(selected)),
    }


def split_indices(
    text_df: pd.DataFrame,
    source_len: int,
    seq_len: int,
    split_mode: str,
    data_name: str = "",
    max_samples: int | None = None,
    train_ratio: float = 1.0,
    train_ratio_seed: int = 2026,
    train_ratio_mode: str = "uniform",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    if max_samples is not None or split_mode == "ratio":
        train, val, test, info = ratio_split_indices(len(text_df), max_samples=max_samples)
        train, ratio_info = _apply_train_ratio(train, train_ratio, train_ratio_seed, train_ratio_mode)
        info.update(ratio_info)
        return train, val, test, info
    if split_mode != "ett_standard":
        raise ValueError(f"Unsupported split mode: {split_mode}")

    points_per_hour = 4 if str(data_name).lower().startswith("ettm") else 1
    train_size = 12 * 30 * 24 * points_per_hour
    val_size = 4 * 30 * 24 * points_per_hour
    test_size = 4 * 30 * 24 * points_per_hour
    train_end = min(train_size, source_len)
    val_end = min(train_size + val_size, source_len)
    test_end = min(train_size + val_size + test_size, source_len)

    start = text_df["start_idx"].to_numpy()
    pred_end = text_df["pred_end_idx"].to_numpy()

    def select(border1: int, border2: int) -> np.ndarray:
        mask = (start >= max(0, border1)) & (pred_end < border2)
        return np.flatnonzero(mask).astype(np.int64)

    train = select(0, train_end)
    val = select(train_end - seq_len, val_end)
    test = select(val_end - seq_len, test_end)
    info = {
        "mode": "ett_standard",
        "train_end": int(train_end),
        "val_end": int(val_end),
        "test_end": int(test_end),
        "points_per_hour": int(points_per_hour),
        "ignored_tail_rows": int(max(0, source_len - test_end)),
    }
    train, ratio_info = _apply_train_ratio(train, train_ratio, train_ratio_seed, train_ratio_mode)
    info.update(ratio_info)
    return train, val, test, info


class CAREDataset(Dataset):
    def __init__(self, args, flag: str):
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported flag: {flag}")
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.features = args.features
        self.student_text_col = args.student_text_col
        self.student_text_cols = list(getattr(args, "student_text_cols", None) or [self.student_text_col])
        self.teacher_text_col = args.teacher_text_col
        self.teacher_text_cols = list(getattr(args, "teacher_text_cols", None) or [self.teacher_text_col])
        self.text_dim = args.text_dim

        source_path = Path(args.root_path) / args.data_path
        text_path = Path(args.text_path)
        if not text_path.is_absolute():
            text_path = Path(args.root_path) / text_path
        if not source_path.exists():
            raise FileNotFoundError(f"Data CSV not found: {source_path}")
        if not text_path.exists():
            raise FileNotFoundError(f"Text CSV not found: {text_path}")

        self.raw_df = pd.read_csv(source_path)
        self.text_df = pd.read_csv(text_path)
        self.numeric_cols = [col for col in self.raw_df.columns if col != "date"]
        if args.target not in self.numeric_cols:
            raise ValueError(f"Target {args.target!r} not found in {self.numeric_cols}")
        self.target_idx = self.numeric_cols.index(args.target)

        train_idx, val_idx, test_idx, self.split_info = split_indices(
            self.text_df,
            source_len=len(self.raw_df),
            seq_len=self.seq_len,
            split_mode=args.split_mode,
            data_name=getattr(args, "data", ""),
            max_samples=args.max_samples,
            train_ratio=float(getattr(args, "train_ratio", 1.0)),
            train_ratio_seed=int(getattr(args, "train_ratio_seed", getattr(args, "seed", 2026))),
            train_ratio_mode=str(getattr(args, "train_ratio_mode", "uniform")),
        )
        index_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        self.indices = index_map[flag]
        if len(self.indices) == 0:
            raise ValueError(f"Empty {flag} split.")

        values = self.raw_df[self.numeric_cols].astype(float).to_numpy(dtype=np.float32)
        train_cut = int(self.text_df.iloc[train_idx[-1]]["pred_end_idx"]) + 1 if len(train_idx) else self.seq_len
        self.scaler = StandardScaler()
        self.scaler.fit(values[:train_cut])
        self.data_x = self.scaler.transform(values).astype(np.float32) if args.scale else values.astype(np.float32)

        feature_path_value = getattr(args, "text_feature_path", "")
        if feature_path_value:
            feature_path = Path(feature_path_value)
            if not feature_path.is_absolute():
                feature_path = Path(args.root_path) / feature_path
            if not feature_path.exists():
                raise FileNotFoundError(f"Text feature NPZ not found: {feature_path}")
            self.text_features = load_text_features_from_npz(self.text_df, feature_path, args.text_dim)
        else:
            cache_key = str(text_path.resolve())
            self.text_features = build_text_features(self.text_df, args.text_dim, args.text_encoder, cache_key)
        for required_col in [*self.student_text_cols, *self.teacher_text_cols, *UNRELIABLE_COLUMNS]:
            if required_col not in self.text_features:
                raise ValueError(f"Missing text feature column: {required_col}")

    @property
    def c_out(self) -> int:
        return len(self.numeric_cols) if self.features == "M" else 1

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        row_idx = int(self.indices[item])
        row = self.text_df.iloc[row_idx]
        start = int(row["start_idx"])
        pred_start = int(row["pred_start_idx"])
        seq_x = self.data_x[start : start + self.seq_len]
        if self.features == "M":
            seq_y = self.data_x[pred_start : pred_start + self.pred_len]
        else:
            seq_y = self.data_x[pred_start : pred_start + self.pred_len, self.target_idx : self.target_idx + 1]

        noise_col = random.choice(UNRELIABLE_COLUMNS) if self.flag == "train" else "irrelevant_text"
        history_parts = [self.text_features[col][row_idx] for col in self.student_text_cols]
        history_text = history_parts[0] if len(history_parts) == 1 else np.concatenate(history_parts, axis=-1)
        teacher_parts = [self.text_features[col][row_idx] for col in self.teacher_text_cols]
        teacher_text = teacher_parts[0] if len(teacher_parts) == 1 else np.concatenate(teacher_parts, axis=-1)
        noise_parts = [self.text_features[noise_col][row_idx] for _ in self.student_text_cols]
        noise_text = noise_parts[0] if len(noise_parts) == 1 else np.concatenate(noise_parts, axis=-1)
        return (
            torch.from_numpy(seq_x.astype(np.float32)),
            torch.from_numpy(seq_y.astype(np.float32)),
            torch.from_numpy(history_text.astype(np.float32)),
            torch.from_numpy(teacher_text.astype(np.float32)),
            torch.from_numpy(noise_text.astype(np.float32)),
            torch.tensor(row_idx, dtype=torch.long),
        )

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.features == "M":
            return self.scaler.inverse_transform(data)
        mean = self.scaler.mean[self.target_idx]
        std = self.scaler.std[self.target_idx]
        return data * std + mean
