from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import SUMMARY_FOR_TEXT, TEXT_COLUMNS, build_text_features


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    arrays = np.load(path)
    return {key: arrays[key].astype(np.float32) for key in arrays.files}


def _normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norm, 1e-6)).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Concatenate cached text feature NPZ files and optional hybrid features.")
    parser.add_argument("--text_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--npz", nargs="*", default=[], help="Input NPZ files to concatenate in order.")
    parser.add_argument("--include_hybrid", action="store_true", help="Append deterministic hybrid text features.")
    parser.add_argument("--hybrid_dim", type=int, default=256)
    parser.add_argument("--columns", nargs="+", default=["all"], help="Columns to export, or 'all' for every known text column.")
    parser.add_argument("--normalize", action="store_true", default=True)
    args = parser.parse_args()

    text_csv = Path(args.text_csv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text_df = pd.read_csv(text_csv)

    sources: list[dict[str, np.ndarray]] = []
    source_dims: list[int] = []
    for value in args.npz:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        source = _load_npz(path)
        sources.append(source)
        source_dims.append(next(iter(source.values())).shape[1])

    if args.include_hybrid:
        cache_key = f"{text_csv.resolve()}::combine_hybrid"
        sources.append(build_text_features(text_df, args.hybrid_dim, "hybrid", cache_key))
        source_dims.append(args.hybrid_dim)

    if not sources:
        raise ValueError("At least one --npz file or --include_hybrid is required.")

    export_columns = TEXT_COLUMNS if args.columns == ["all"] else args.columns
    out: dict[str, np.ndarray] = {}
    for col in export_columns:
        if col not in text_df.columns:
            continue
        blocks: list[np.ndarray] = []
        has_any = False
        for source, dim in zip(sources, source_dims):
            if col in source:
                block = source[col]
                has_any = True
            else:
                block = np.zeros((len(text_df), dim), dtype=np.float32)
            blocks.append(block)
        if not has_any:
            print(f"skip missing column: {col}")
            continue
        rows = {block.shape[0] for block in blocks}
        if rows != {len(text_df)}:
            raise ValueError(f"{col} row mismatch: {sorted(rows)} vs text rows {len(text_df)}")
        features = np.concatenate(blocks, axis=1).astype(np.float32)
        if args.normalize:
            features = _normalize(features)
        out[col] = features
        print(f"{col}: {features.shape}")

    # Preserve zero placeholders if a downstream run requests a missing column.
    dim = next(iter(out.values())).shape[1]
    zero = np.zeros((len(text_df), dim), dtype=np.float32)
    for col in TEXT_COLUMNS:
        if col in text_df.columns and col not in out and col not in SUMMARY_FOR_TEXT:
            out[col] = zero.copy()

    np.savez_compressed(output, **out)
    print(f"wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
