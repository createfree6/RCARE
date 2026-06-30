from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

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


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def encode_column(texts: list[str], tokenizer, model, device: torch.device, batch_size: int, max_length: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        hidden = model(**tokens).last_hidden_state
        pooled = mean_pool(hidden, tokens["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        outputs.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute frozen transformer text embeddings for CARE-Forecast.")
    parser.add_argument("--text_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--columns", nargs="+", default=["llm_history_text", "llm_residual_text"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    text_csv = Path(args.text_csv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(text_csv)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True).to(args.device)
    model.eval()

    arrays: dict[str, np.ndarray] = {}
    for col in args.columns:
        if col not in df.columns:
            print(f"skip missing column: {col}")
            continue
        texts = df[col].fillna("No textual information available.").astype(str).tolist()
        print(f"encoding {col}: {len(texts)} rows")
        arrays[col] = encode_column(texts, tokenizer, model, torch.device(args.device), args.batch_size, args.max_length)
        print(f"  shape={arrays[col].shape}")

    np.savez_compressed(output, **arrays)
    print(f"wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
