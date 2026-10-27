"""
Publication-grade visualizations for experimental results.
Generates:
1. accuracy_comparison.png: Exact call & tool selection accuracy by condition
2. token_comparison.png: Mean & median input tokens across conditions
3. shortlist_recall.png: Semantic top-3 tool shortlist recall
4. task_level_comparison.png: Paired task-by-task correctness matrix
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, RESULTS_DIR


# Clean scientific aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def plot_accuracy_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """Plot Tool Selection Accuracy and Exact Call Accuracy across the 3 conditions."""
    out_file = output_path or (FIGURES_DIR / "accuracy_comparison.png")

    conditions = ["Condition_A", "Condition_B", "Condition_C"]
    cond_labels = [
        "Condition A\n(Original + Full 20)",
        "Condition B\n(Paraphrased + Full 20)",
        "Condition C\n(Paraphrased + Top-3 Shortlist)"
    ]

    tool_accs = []
    exact_accs = []
    n_samples = []

    for c in conditions:
        subset = df[df["condition"] == c]
        n_samples.append(len(subset))
        if len(subset) > 0:
            tool_accs.append(float(subset["tool_name_correct"].mean()) * 100.0)
            exact_accs.append(float(subset["exact_call_correct"].mean()) * 100.0)
        else:
            tool_accs.append(0.0)
            exact_accs.append(0.0)

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, tool_accs, width, label="Tool Selection Accuracy", color="#3b82f6", edgecolor="#1d4ed8", alpha=0.9)
    rects2 = ax.bar(x + width/2, exact_accs, width, label="Exact Function Call Accuracy", color="#10b981", edgecolor="#047857", alpha=0.9)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy Across Experimental Conditions (N={n_samples[0]} Paired Tasks)")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True)

    # Bar labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_token_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """Plot Mean and Median Input Token Consumption by Condition."""
    out_file = output_path or (FIGURES_DIR / "token_comparison.png")

    conditions = ["Condition_A", "Condition_B", "Condition_C"]
    cond_labels = [
        "Condition A\n(Full 20 Tools)",
        "Condition B\n(Full 20 Tools)",
        "Condition C\n(Top-3 Shortlist)"
    ]

    means = []
    medians = []

    for c in conditions:
        tokens = df[df["condition"] == c]["input_token_count"].dropna()
        if len(tokens) > 0:
            means.append(float(tokens.mean()))
            medians.append(float(tokens.median()))
        else:
            means.append(0.0)
            medians.append(0.0)

    # Savings calculation
    savings_pct = 0.0
    if means[1] > 0:
        savings_pct = ((means[1] - means[2]) / means[1]) * 100.0

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    r1 = ax.bar(x - width/2, means, width, label="Mean Input Tokens", color="#8b5cf6", edgecolor="#6d28d9", alpha=0.9)
    r2 = ax.bar(x + width/2, medians, width, label="Median Input Tokens", color="#ec4899", edgecolor="#be185d", alpha=0.9)

    ax.set_ylabel("Input Token Count")
    ax.set_title(f"Input Token Cost Comparison (Condition C Savings: ~{savings_pct:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylim(0, max(means + medians) * 1.25 if means else 200)
    ax.legend(loc="upper right", frameon=True)

    for rect in r1:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    for rect in r2:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_shortlist_recall(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """Plot Shortlist Recall for Condition C."""
    out_file = output_path or (FIGURES_DIR / "shortlist_recall.png")

    df_c = df[df["condition"] == "Condition_C"]
    valid = df_c[df_c["shortlist_recall"] >= 0]
    recall_rate = float(valid["shortlist_recall"].mean()) * 100.0 if len(valid) > 0 else 0.0
    miss_rate = 100.0 - recall_rate

    fig, ax = plt.subplots(figsize=(6, 5))
    categories = ["Target Tool Retrieved\n(in Top-3)", "Target Tool Missed"]
    values = [recall_rate, miss_rate]
    colors = ["#10b981", "#ef4444"]

    bars = ax.bar(categories, values, color=colors, width=0.45, edgecolor="#1f2937")
    ax.set_ylabel("Percentage of Tasks (%)")
    ax.set_title(f"Condition C: Top-3 Shortlist Recall Rate (N={len(valid)} Tasks)")
    ax.set_ylim(0, 115)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_task_level_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """Plot paired task-by-task exact call correctness matrix."""
    out_file = output_path or (FIGURES_DIR / "task_level_comparison.png")

    pivot = df.pivot(index="task_id", columns="condition", values="exact_call_correct")
    # Ensure standard order
    cols = [c for c in ["Condition_A", "Condition_B", "Condition_C"] if c in pivot.columns]
    pivot = pivot[cols].astype(int)

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        pivot,
        annot=True,
        cmap=["#fca5a5", "#86efac"],
        cbar=False,
        linewidths=1.0,
        linecolor="#e5e7eb",
        ax=ax,
        fmt="d"
    )
    ax.set_title("Paired Task-Level Exact Call Success (1=Correct, 0=Incorrect)")
    ax.set_xlabel("Experimental Condition")
    ax.set_ylabel("Benchmark Task ID")
    ax.set_xticklabels(["Condition A\n(Orig + Full)", "Condition B\n(Para + Full)", "Condition C\n(Para + Shortlist)"])

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_failure_breakdown(
    df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> Path:
    """Plot breakdown of failure modes in Condition C."""
    out_file = output_path or (FIGURES_DIR / "failure_breakdown.png")

    df_c = df[df["condition"] == "Condition_C"]
    total = len(df_c)
    if total == 0:
        return out_file

    retriever_misses = 0
    llm_tool_errors = 0
    arg_errors = 0
    api_errors = 0
    exact_successes = 0

    for _, row in df_c.iterrows():
        if not row.get("api_success", True):
            api_errors += 1
        elif row.get("exact_call_correct", False):
            exact_successes += 1
        elif row.get("shortlist_recall", 1) == 0:
            retriever_misses += 1
        elif not row.get("tool_name_correct", False):
            llm_tool_errors += 1
        else:
            arg_errors += 1

    categories = [
        "Exact Call\nSuccess",
        "Retriever Miss\n(Absent from Top-3)",
        "LLM Tool Error\n(Wrong Tool Selected)",
        "Argument Mismatch\n(Correct Tool)",
        "API / Parse\nFailure"
    ]
    counts = [exact_successes, retriever_misses, llm_tool_errors, arg_errors, api_errors]
    colors = ["#10b981", "#ef4444", "#f59e0b", "#3b82f6", "#6b7280"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, counts, color=colors, edgecolor="#1f2937", width=0.5)

    ax.set_ylabel("Number of Tasks")
    ax.set_title(f"Condition C Performance Breakdown (N={total} Tasks)")
    ax.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 10)

    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100.0 if total > 0 else 0.0
        ax.annotate(f"{h} ({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def generate_all_figures(df: pd.DataFrame) -> None:
    """Generate all required publication figures."""
    plot_accuracy_comparison(df)
    plot_token_comparison(df)
    plot_shortlist_recall(df)
    plot_task_level_comparison(df)
    plot_failure_breakdown(df)
