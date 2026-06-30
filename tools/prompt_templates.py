from __future__ import annotations

import json
import re
from typing import Any


SYSTEM_PROMPT = (
    "You are a careful time-series analyst. Your job is to convert statistical summaries "
    "of a time-series window into concise textual context for a forecasting model. "
    "Do not hallucinate exact future values or external events. Use only the supplied summary. "
    "Return valid JSON only."
)


def _json_block(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _allowed_variables(row: dict[str, Any], target: str) -> str:
    values = {target}
    for key in ["positive_variables", "negative_variables"]:
        values.update(_parse_json_list(row.get(key)))
    evidence = " ".join(str(row.get(key, "")) for key in ["history_text", "future_text", "residual_text"])
    values.update(re.findall(r"\b[A-Z][A-Z0-9_]{1,10}\b", evidence))
    return ", ".join(sorted(item for item in values if item and item.lower() != "nan"))


def _anti_hallucination_rules(row: dict[str, Any], target: str) -> str:
    allowed = _allowed_variables(row, target)
    return f"""Anti-hallucination rules:
- Allowed variable names: {allowed}.
- Do not introduce external domain nouns such as sales, employment, inflation, policy, stock, market, traffic, weather, price, or demand unless they appear verbatim in the evidence.
- Do not write exact numeric values, signed values such as +0.8, units, or literal future targets.
- If a variable is not named in the evidence, write \"the target variable\" or \"auxiliary variables\" instead."""


def build_history_prompt(row: dict[str, Any], target: str, features: str = "M") -> str:
    scope = "all variables jointly" if features == "M" else f"the target variable {target}"
    return f"""Task: summarize the observed historical time-series window for {scope}.

Strict rules:
1. Use only the observed history from {row['start_date']} to {row['end_date']}.
2. Do not mention the future segment or any value after {row['end_date']}.
3. Describe trend, volatility, recent change, periodicity, anomaly evidence, uncertainty, and cross-variable relations when available.
4. Avoid exact numeric forecasts. Keep the summary useful for a forecasting model.
5. Explicitly state whether the text is high-confidence, medium-confidence, or low-confidence for forecasting.
6. Return JSON with keys: trend, volatility, recent_change, periodicity, anomaly, uncertainty, cross_variable_context, forecast_relevance, confidence, concise_text.

{_anti_hallucination_rules(row, target)}

Historical summary JSON:
{_json_block(row['history_summary'])}

Positive related variables: {row.get('positive_variables', '[]')}
Negative related variables: {row.get('negative_variables', '[]')}

Write `concise_text` as one complete paragraph under 70 words."""


def build_compact_history_prompt(row: dict[str, Any], target: str, features: str = "M") -> str:
    scope = "all variables jointly" if features == "M" else f"the target variable {target}"
    return f"""Task: rewrite the supplied time-series history evidence into a compact forecasting context for {scope}.

Strict rules:
1. Use only the observed history evidence.
2. Do not mention or infer the future segment.
3. Preserve trend, volatility, anomaly, uncertainty, and cross-variable clues from the evidence.
4. Do not include exact numeric values or external events.
5. Return valid JSON only with keys: trend, volatility, anomaly, uncertainty, forecast_relevance, confidence, concise_text.

{_anti_hallucination_rules(row, target)}

Observed-history evidence:
{row.get('history_text', '')}

Compact tags:
{row.get('compact_text', '')}

Write `concise_text` as one complete sentence under 35 words."""


def build_history_future_prompt(row: dict[str, Any], target: str, features: str = "M") -> str:
    scope = "all variables jointly" if features == "M" else f"the target variable {target}"
    return f"""Task: infer a deployable future-response prior for {scope} using only the observed history.

Strict rules:
1. Use only information available in the observed history; do not use the true future segment.
2. Describe the likely next-window response qualitatively as a prior, not as a guaranteed outcome.
3. Preserve trend, recent slope, uncertainty, mean-reversion risk, and historical driver clues from the evidence.
4. Do not include exact values, timestamps, or external events.
5. Return valid JSON only with keys: inferred_response, uncertainty, driver_context, mean_reversion_risk, confidence, concise_text.

{_anti_hallucination_rules(row, target)}

History-only predictive evidence:
{row.get('llm_history_future_text', row.get('history_text', ''))}

History-future summary JSON:
{_json_block(row.get('history_future_summary', row.get('history_summary', '{}')))}

Write `concise_text` as one complete sentence under 35 words."""


def build_future_prompt(row: dict[str, Any], target: str, features: str = "M") -> str:
    scope = "all variables jointly" if features == "M" else f"the target variable {target}"
    return f"""Task: summarize a privileged future pattern for {scope}.

Strict rules:
1. This text is training-only privileged information for a teacher model.
2. Use only the provided future summary from {row['pred_start_date']} to {row['pred_end_date']}.
3. Do not list exact future values. Describe qualitative direction, volatility, anomalies, uncertainty, and temporal impact.
4. If a delayed or lagged effect is visible in the supplied summary, describe it qualitatively; otherwise write "not evident".
5. Make the summary actionable as teacher guidance, not as a direct numeric answer.
6. Return JSON with keys: future_trend, future_volatility, future_change, future_anomaly, future_uncertainty, delayed_effect_hint, confidence, teacher_hint, concise_text.

{_anti_hallucination_rules(row, target)}

Future summary JSON:
{_json_block(row['future_summary'])}

Write `concise_text` as one complete paragraph under 70 words."""


def build_compact_future_prompt(row: dict[str, Any], target: str, features: str = "M") -> str:
    scope = "all variables jointly" if features == "M" else f"the target variable {target}"
    return f"""Task: rewrite the supplied privileged future evidence into compact teacher guidance for {scope}.

Strict rules:
1. This is training-only privileged text for a teacher model.
2. Use only the supplied future evidence.
3. Describe qualitative direction, volatility, uncertainty, anomaly risk, and delayed-effect hint if evident.
4. Do not include exact target values or external events.
5. Return valid JSON only with keys: future_trend, future_volatility, future_uncertainty, anomaly_risk, delayed_effect_hint, confidence, concise_text.

{_anti_hallucination_rules(row, target)}

Privileged future evidence:
{row.get('future_text', '')}

Write `concise_text` as one complete sentence under 35 words."""


def build_residual_prompt(row: dict[str, Any], target: str, features: str = "M") -> str:
    scope = "all variables jointly" if features == "M" else f"the target variable {target}"
    return f"""Task: summarize the future residual correction for {scope} relative to a persistence baseline.

Strict rules:
1. This text is training-only privileged information for a teacher model.
2. The residual is future value minus the last observed value repeated as a naive baseline.
3. Focus on correction direction, correction strength, volatility, uncertainty, and which variables need stronger correction.
4. Do not reveal exact numeric targets. Do not invent external events.
5. If the residual suggests delayed impact beyond the observed history, describe it qualitatively; otherwise write "not evident".
6. Return JSON with keys: correction_direction, correction_strength, residual_volatility, residual_uncertainty, variables_requiring_correction, lagged_effect_hint, confidence, teacher_hint, concise_text.

{_anti_hallucination_rules(row, target)}

Residual summary JSON:
{_json_block(row['residual_summary'])}

Write `concise_text` as one complete paragraph under 70 words."""


def build_compact_residual_prompt(row: dict[str, Any], target: str, features: str = "M") -> str:
    scope = "all variables jointly" if features == "M" else f"the target variable {target}"
    return f"""Task: rewrite the supplied residual evidence into compact teacher-only correction guidance for {scope}.

Strict rules:
1. This is training-only privileged text for a teacher model.
2. The residual means future value minus the last-observed-value persistence baseline.
3. Preserve qualitative correction direction, correction strength, volatility, uncertainty, and variables needing correction.
4. Do not include exact future target values or external events.
5. Return valid JSON only with keys: correction_direction, correction_strength, residual_volatility, residual_uncertainty, variables_requiring_correction, confidence, concise_text.

{_anti_hallucination_rules(row, target)}

Privileged residual evidence:
{row.get('residual_text', '')}

Write `concise_text` as one complete sentence under 35 words."""


def build_prompt(row: dict[str, Any], kind: str, target: str = "OT", features: str = "M") -> str:
    if kind == "history":
        return build_history_prompt(row, target=target, features=features)
    if kind == "history_future":
        return build_history_future_prompt(row, target=target, features=features)
    if kind == "future":
        return build_future_prompt(row, target=target, features=features)
    if kind == "residual":
        return build_residual_prompt(row, target=target, features=features)
    raise ValueError(f"Unsupported prompt kind: {kind}")


def build_compact_prompt(row: dict[str, Any], kind: str, target: str = "OT", features: str = "M") -> str:
    if kind == "history":
        return build_compact_history_prompt(row, target=target, features=features)
    if kind == "history_future":
        return build_history_future_prompt(row, target=target, features=features)
    if kind == "future":
        return build_compact_future_prompt(row, target=target, features=features)
    if kind == "residual":
        return build_compact_residual_prompt(row, target=target, features=features)
    raise ValueError(f"Unsupported prompt kind: {kind}")
