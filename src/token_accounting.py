"""
Token accounting, provider metadata extraction, and fallback estimation.
Separates provider-reported tokens from heuristic estimates.
"""

from typing import Any, Dict, Optional, Tuple


def extract_token_counts(
    response_json: Optional[Dict[str, Any]],
    query_text: str = "",
    tools_payload: Optional[list] = None
) -> Tuple[int, int, str]:
    """
    Extract token usage from Gemini usageMetadata if present.
    If absent, compute a documented heuristic estimate (approx. 4 characters per token).

    Returns:
        (input_token_count, output_token_count, token_count_source)
        where token_count_source is either 'provider_reported' or 'heuristic_estimated'.
    """
    if response_json and "usageMetadata" in response_json:
        usage = response_json["usageMetadata"]
        prompt_tokens = usage.get("promptTokenCount", 0)
        candidates_tokens = usage.get("candidatesTokenCount", 0)
        if prompt_tokens > 0 or candidates_tokens > 0:
            return prompt_tokens, candidates_tokens, "provider_reported"

    # Documented fallback heuristic: 1 token ~= 4 characters for English text & JSON schema
    import json
    text_len = len(query_text)
    tools_str = json.dumps(tools_payload or [])
    tools_len = len(tools_str)

    est_input_tokens = max(1, (text_len + tools_len) // 4)
    est_output_tokens = 20  # standard typical tool call response size

    return est_input_tokens, est_output_tokens, "heuristic_estimated"


def compute_token_savings(
    full_tokens: float,
    shortlist_tokens: float
) -> float:
    """
    Compute percentage savings in input token consumption.
    formula: ((full - shortlist) / full) * 100.0
    """
    if full_tokens <= 0:
        return 0.0
    savings = ((full_tokens - shortlist_tokens) / full_tokens) * 100.0
    return max(0.0, min(100.0, savings))
