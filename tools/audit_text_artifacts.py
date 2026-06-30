from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.data_loader import split_indices


SUSPICIOUS_HISTORY_PATTERNS = [
    r"\bfuture\b",
    r"\bteacher\b",
    r"\bprivileged\b",
    r"\bresidual\b",
    r"\bprediction segment\b",
    r"\bforecast horizon\b",
]

HALLUCINATED_DOMAIN_PATTERNS = [
    r"\bsales\b",
    r"\bemployment\b",
    r"\binflation\b",
    r"\bpolicy\b",
    r"\bstock\b",
    r"\bmarket\b",
    r"\beconomy\b",
    r"\btraffic\b",
    r"\bweather\b",
    r"\bprice\b",
    r"\bdemand\b",
    r"[+-]\d+(\.\d+)?",
]


def _is_nonempty(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def _length_stats(series: pd.Series) -> tuple[float, int, int]:
    if len(series) == 0:
        return 0.0, 0, 0
    lengths = series.fillna("").astype(str).str.split().map(len)
    return float(lengths.mean()), int(lengths.min()), int(lengths.max())


def _suspicious_rate(series: pd.Series, patterns: list[str]) -> tuple[int, list[str]]:
    regexes = [re.compile(pattern, flags=re.I) for pattern in patterns]
    hits: list[str] = []
    count = 0
    for text in series.fillna("").astype(str).tolist():
        matched = [pattern.pattern for pattern in regexes if pattern.search(text)]
        if matched:
            count += 1
            if len(hits) < 3:
                hits.append(text[:220].replace("\n", " "))
    return count, hits


def _unfinished_rate(series: pd.Series) -> tuple[int, list[str]]:
    hits: list[str] = []
    count = 0
    for text in series.fillna("").astype(str).tolist():
        stripped = text.strip()
        if not stripped or stripped.lower() in {"nan", "none", "null"}:
            continue
        if stripped[-1] not in ".!?;:":
            count += 1
            if len(hits) < 3:
                hits.append(stripped[:220].replace("\n", " "))
    return count, hits


def _split_for_df(
    df: pd.DataFrame,
    data_csv: Path | None,
    data_name: str,
    seq_len: int,
    split_mode: str,
) -> tuple[list[str], list[int]]:
    if data_csv is None or not data_csv.exists():
        return ["Data CSV not provided; split audit skipped."], list(range(len(df)))
    source_len = len(pd.read_csv(data_csv, usecols=[0]))
    train, val, test, info = split_indices(df, source_len, seq_len, split_mode, data_name=data_name, max_samples=None)
    used = sorted(set(train.tolist()) | set(val.tolist()) | set(test.tolist()))
    return [
        "| split | windows |",
        "|---|---:|",
        f"| train | {len(train)} |",
        f"| val | {len(val)} |",
        f"| test | {len(test)} |",
        f"| used union | {len(used)} |",
        "",
        f"Split info: `{info}`",
    ], used


def build_report(args: argparse.Namespace) -> tuple[str, bool]:
    text_csv = Path(args.text_csv)
    df = pd.read_csv(text_csv)
    data_csv = Path(args.data_csv) if args.data_csv else None
    split_lines, used_indices = _split_for_df(df, data_csv, args.data_name, args.seq_len, args.split_mode)
    if args.coverage_scope == "used" and data_csv is not None and data_csv.exists():
        check_df = df.iloc[used_indices].copy()
        scope_name = "used train/val/test windows"
    else:
        check_df = df
        scope_name = "all rows"
    lines: list[str] = [
        f"# Text Artifact Audit: {text_csv.name}",
        "",
        "This audit checks whether cached LLM text can be used as a reproducible offline modality.",
        "",
        "## Dataset",
        "",
        f"- rows: {len(df)}",
        f"- coverage scope: {scope_name} ({len(check_df)} rows checked)",
        f"- columns: {len(df.columns)}",
        f"- csv: `{text_csv}`",
        "",
        "## Split Check",
        "",
        *split_lines,
        "",
        "## Required Text Columns",
        "",
        "| column | exists | nonempty | empty | mean_words | min_words | max_words | duplicate_rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ok = True
    for col in args.required_columns:
        exists = col in df.columns
        if not exists:
            lines.append(f"| {col} | no | 0 | {len(check_df)} | - | - | - | - |")
            ok = False
            continue
        nonempty_mask = check_df[col].map(_is_nonempty)
        nonempty = int(nonempty_mask.sum())
        empty = int(len(check_df) - nonempty)
        if args.require_full and empty:
            ok = False
        mean_len, min_len, max_len = _length_stats(check_df.loc[nonempty_mask, col] if nonempty else pd.Series([], dtype=str))
        dup_rate = float(check_df.loc[nonempty_mask, col].duplicated().mean()) if nonempty else 0.0
        lines.append(
            f"| {col} | yes | {nonempty} | {empty} | {mean_len:.1f} | {min_len} | {max_len} | {dup_rate:.3f} |"
        )

    lines.extend(["", "## Leakage / Prompt-Compliance Heuristics", ""])
    if "llm_history_text" in check_df.columns:
        history_series = check_df.loc[check_df["llm_history_text"].map(_is_nonempty), "llm_history_text"]
        count, examples = _suspicious_rate(history_series, SUSPICIOUS_HISTORY_PATTERNS)
        rate = count / max(len(history_series), 1)
        if rate > args.max_history_suspicious_rate:
            ok = False
        lines.extend(
            [
                f"- `llm_history_text` suspicious keyword hits: {count}/{len(history_series)} ({rate:.3%}).",
                "- This is a heuristic only; inspect examples before making a claim.",
            ]
        )
        if examples:
            lines.append("")
            lines.append("Examples:")
            for item in examples:
                lines.append(f"- {item}")
    else:
        lines.append("- `llm_history_text` not present; history leakage audit skipped.")

    for col in [item for item in args.required_columns if item in check_df.columns]:
        series = check_df.loc[check_df[col].map(_is_nonempty), col]
        count, examples = _suspicious_rate(series, HALLUCINATED_DOMAIN_PATTERNS)
        rate = count / max(len(series), 1)
        if rate > args.max_hallucination_rate:
            ok = False
        lines.append(f"- `{col}` hallucinated-domain/exact-number hits: {count}/{len(series)} ({rate:.3%}).")
        if examples:
            lines.append("")
            lines.append(f"{col} hallucination examples:")
            for item in examples:
                lines.append(f"- {item}")
        unfinished, unfinished_examples = _unfinished_rate(series)
        unfinished_rate = unfinished / max(len(series), 1)
        if unfinished_rate > args.max_unfinished_rate:
            ok = False
        lines.append(f"- `{col}` unfinished-sentence hits: {unfinished}/{len(series)} ({unfinished_rate:.3%}).")
        if unfinished_examples:
            lines.append("")
            lines.append(f"{col} unfinished examples:")
            for item in unfinished_examples:
                lines.append(f"- {item}")

    lines.extend(["", "## Text Previews", ""])
    preview_cols = [col for col in args.required_columns if col in check_df.columns]
    for idx in range(min(args.preview_rows, len(check_df))):
        row = check_df.iloc[idx]
        lines.append(f"### sample_id={row.get('sample_id', idx)}")
        for col in preview_cols:
            text = str(row.get(col, "")).strip().replace("\n", " ")
            lines.append(f"- {col}: {text[:260]}")
        lines.append("")

    lines.extend(
        [
            "## Verdict",
            "",
            "PASS"
            if ok
            else "INSPECT: coverage or quality thresholds failed under the selected strictness.",
            "",
        ]
    )
    return "\n".join(lines), ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit cached LLM text artifacts for RCARE-Forecast.")
    parser.add_argument("--text-csv", required=True)
    parser.add_argument("--data-csv", default="")
    parser.add_argument("--output", default="outputs/text_artifact_audit.md")
    parser.add_argument("--required-columns", nargs="+", default=["llm_history_text", "llm_residual_text"])
    parser.add_argument("--require-full", action="store_true", help="Fail if any required column has empty rows.")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--split-mode", choices=["ett_standard", "ratio"], default="ett_standard")
    parser.add_argument("--data-name", default="")
    parser.add_argument(
        "--coverage-scope",
        choices=["used", "all"],
        default="used",
        help="Rows checked by --require-full and quality heuristics. 'used' means train/val/test split rows.",
    )
    parser.add_argument("--max-unfinished-rate", type=float, default=1.0)
    parser.add_argument("--max-hallucination-rate", type=float, default=1.0)
    parser.add_argument("--max-history-suspicious-rate", type=float, default=1.0)
    parser.add_argument("--preview-rows", type=int, default=3)
    args = parser.parse_args()

    report, ok = build_report(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(output)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
