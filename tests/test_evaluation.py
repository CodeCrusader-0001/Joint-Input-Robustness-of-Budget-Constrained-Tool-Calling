"""
Unit tests for evaluation metrics, argument comparison, and statistical analysis.
"""

import numpy as np
import pandas as pd
import pytest
from src.evaluation import (
    compute_bootstrap_ci,
    compute_condition_metrics,
    compute_paired_comparison,
)
from src.function_calling import compare_arguments


def test_compare_arguments_exact():
    predicted = {"city": "London", "units": "celsius"}
    expected = {"city": "London", "units": "celsius"}
    correct, details = compare_arguments(predicted, expected)
    assert correct is True
    assert len(details) == 0


def test_compare_arguments_numeric_tolerance():
    # 150 vs 150.0 should be matched
    predicted = {"amount": 150.0, "from_currency": "USD", "to_currency": "EUR"}
    expected = {"amount": 150, "from_currency": "USD", "to_currency": "EUR"}
    correct, details = compare_arguments(predicted, expected)
    assert correct is True


def test_compare_arguments_case_insensitivity():
    predicted = {"symbol": "aapl"}
    expected = {"symbol": "AAPL"}
    correct, details = compare_arguments(predicted, expected)
    assert correct is True


def test_compare_arguments_missing_and_wrong():
    predicted = {"city": "Paris"}
    expected = {"city": "London", "units": "celsius"}
    correct, details = compare_arguments(predicted, expected)
    assert correct is False
    assert "units" in details
    assert details["units"]["status"] == "missing"
    assert "city" in details
    assert details["city"]["status"] == "string_mismatch"


def test_compute_condition_metrics():
    df = pd.DataFrame([
        {
            "condition": "Condition_A",
            "tool_name_correct": True,
            "arguments_correct": True,
            "exact_call_correct": True,
            "api_success": True,
            "input_token_count": 120,
            "latency_seconds": 1.2
        },
        {
            "condition": "Condition_A",
            "tool_name_correct": True,
            "arguments_correct": False,
            "exact_call_correct": False,
            "api_success": True,
            "input_token_count": 130,
            "latency_seconds": 1.4
        }
    ])
    metrics = compute_condition_metrics(df)
    assert metrics["evaluable_tasks"] == 2
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["argument_accuracy"] == 0.5
    assert metrics["exact_call_accuracy"] == 0.5
    assert metrics["mean_input_tokens"] == 125.0


def test_compute_bootstrap_ci():
    data = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
    ci_low, ci_high = compute_bootstrap_ci(data, n_bootstraps=500)
    assert 0.0 <= ci_low <= ci_high <= 1.0


def test_compute_paired_comparison():
    df = pd.DataFrame([
        {"task_id": "t1", "condition": "Cond1", "exact_call_correct": 1},
        {"task_id": "t2", "condition": "Cond1", "exact_call_correct": 1},
        {"task_id": "t1", "condition": "Cond2", "exact_call_correct": 1},
        {"task_id": "t2", "condition": "Cond2", "exact_call_correct": 0},
    ])
    res = compute_paired_comparison(df, "Cond1", "Cond2", metric="exact_call_correct")
    assert res["n_paired"] == 2
    assert res["cond1_mean"] == 1.0
    assert res["cond2_mean"] == 0.5
    assert res["mean_difference"] == 0.5
