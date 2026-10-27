"""
Statistical evaluation, paired comparison metrics, bootstrap confidence intervals,
and uncertainty analysis for the tool-calling experiment.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import RANDOM_SEED


def compute_condition_metrics(df_cond: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregated metrics for a single experimental condition."""
    n_total = len(df_cond)
    if n_total == 0:
        return {
            "evaluable_tasks": 0,
            "tool_selection_accuracy": 0.0,
            "argument_accuracy": 0.0,
            "exact_call_accuracy": 0.0,
            "api_success_rate": 0.0,
            "mean_input_tokens": 0.0,
            "median_input_tokens": 0.0,
            "mean_latency_seconds": 0.0,
            "median_latency_seconds": 0.0
        }

    successful = df_cond[df_cond["api_success"] == True]
    n_success = len(successful)

    tool_acc = float(df_cond["tool_name_correct"].mean())
    arg_acc = float(df_cond["arguments_correct"].mean())
    exact_acc = float(df_cond["exact_call_correct"].mean())
    api_success_rate = float(df_cond["api_success"].mean())

    input_tokens = df_cond["input_token_count"].dropna()
    latencies = df_cond[df_cond["api_success"] == True]["latency_seconds"].dropna()

    return {
        "evaluable_tasks": n_total,
        "successful_api_calls": n_success,
        "api_success_rate": round(api_success_rate, 4),
        "tool_selection_accuracy": round(tool_acc, 4),
        "argument_accuracy": round(arg_acc, 4),
        "exact_call_accuracy": round(exact_acc, 4),
        "mean_input_tokens": round(float(input_tokens.mean()), 2) if len(input_tokens) else 0.0,
        "median_input_tokens": round(float(input_tokens.median()), 2) if len(input_tokens) else 0.0,
        "mean_latency_seconds": round(float(latencies.mean()), 3) if len(latencies) else 0.0,
        "median_latency_seconds": round(float(latencies.median()), 3) if len(latencies) else 0.0
    }


def compute_bootstrap_ci(
    data: np.ndarray,
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    seed: int = RANDOM_SEED
) -> Tuple[float, float]:
    """Compute empirical percentile bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstraps):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return round(lower, 4), round(upper, 4)


def compute_paired_comparison(
    df: pd.DataFrame,
    cond1: str,
    cond2: str,
    metric: str = "exact_call_correct"
) -> Dict[str, Any]:
    """
    Perform paired comparison between two conditions evaluated on the same tasks.
    Calculates paired differences, bootstrap 95% CI, and paired contingency statistics.
    """
    df1 = df[df["condition"] == cond1].sort_values("task_id").reset_index(drop=True)
    df2 = df[df["condition"] == cond2].sort_values("task_id").reset_index(drop=True)

    # Merge on task_id to ensure strict pairing
    merged = pd.merge(df1, df2, on="task_id", suffixes=(f"_{cond1}", f"_{cond2}"))
    n_paired = len(merged)

    if n_paired == 0:
        return {
            "comparison": f"{cond1} vs {cond2}",
            "n_paired": 0,
            "mean_diff": 0.0,
            "ci_95": [0.0, 0.0]
        }

    val1 = merged[f"{metric}_{cond1}"].astype(float).values
    val2 = merged[f"{metric}_{cond2}"].astype(float).values
    diffs = val1 - val2  # positive means cond1 > cond2
    mean_diff = float(np.mean(diffs))
    ci_lower, ci_upper = compute_bootstrap_ci(diffs)

    # Contingency table for binary outcome
    b = int(np.sum((val1 == 1) & (val2 == 0)))  # cond1 correct, cond2 incorrect
    c = int(np.sum((val1 == 0) & (val2 == 1)))  # cond1 incorrect, cond2 correct

    # McNemar's exact binomial p-value for discordant pairs
    if (b + c) > 0:
        # Two-sided binomial test under H0: p=0.5
        mcnemar_p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    else:
        mcnemar_p = 1.0

    return {
        "comparison": f"{cond1} vs {cond2}",
        "n_paired": n_paired,
        "cond1_mean": round(float(np.mean(val1)), 4),
        "cond2_mean": round(float(np.mean(val2)), 4),
        "mean_difference": round(mean_diff, 4),
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "discordant_pairs": {"cond1_only": b, "cond2_only": c},
        "mcnemar_p_value": round(mcnemar_p, 4),
        "statistically_significant_at_05": mcnemar_p < 0.05
    }
