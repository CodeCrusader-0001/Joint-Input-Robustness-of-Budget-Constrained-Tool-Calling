"""
Unit tests for token accounting, provider extraction, and savings calculation.
"""

import pytest
from src.token_accounting import compute_token_savings, extract_token_counts


def test_extract_token_counts_provider():
    raw_resp = {
        "usageMetadata": {
            "promptTokenCount": 245,
            "candidatesTokenCount": 18,
            "totalTokenCount": 263
        }
    }
    in_tok, out_tok, src = extract_token_counts(raw_resp)
    assert in_tok == 245
    assert out_tok == 18
    assert src == "provider_reported"


def test_extract_token_counts_heuristic():
    query = "Find an Italian restaurant in Boston"
    tools_payload = [{"name": "find_restaurant", "description": "finds restaurants"}]
    in_tok, out_tok, src = extract_token_counts(None, query, tools_payload)
    assert in_tok > 0
    assert out_tok > 0
    assert src == "heuristic_estimated"


def test_compute_token_savings():
    full_tokens = 500.0
    shortlist_tokens = 150.0
    savings = compute_token_savings(full_tokens, shortlist_tokens)
    assert savings == 70.0

    # Edge cases
    assert compute_token_savings(0.0, 100.0) == 0.0
    assert compute_token_savings(100.0, 100.0) == 0.0
