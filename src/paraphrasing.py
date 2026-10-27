"""
Paraphrase processing, validation, and lexical divergence metrics.
Ensures identical paraphrased inputs are applied across Condition B and Condition C.
"""

from typing import Dict, List, Tuple
from src.dataset import Task


def compute_jaccard_similarity(str1: str, str2: str) -> float:
    """Compute token-level Jaccard similarity between two query strings."""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def compute_lexical_diversity(tasks: List[Task]) -> Dict[str, float]:
    """
    Compute lexical difference metrics between original and paraphrased queries.
    Confirms that paraphrases introduce real lexical variation while keeping semantic constraints.
    """
    jaccards = []
    char_len_diffs = []

    for t in tasks:
        j = compute_jaccard_similarity(t.original_query, t.paraphrased_query)
        jaccards.append(j)
        char_len_diffs.append(abs(len(t.paraphrased_query) - len(t.original_query)))

    mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
    mean_char_diff = sum(char_len_diffs) / len(char_len_diffs) if char_len_diffs else 0.0

    return {
        "mean_jaccard_similarity": round(mean_jaccard, 4),
        "mean_lexical_distance": round(1.0 - mean_jaccard, 4),
        "mean_character_length_diff": round(mean_char_diff, 2),
        "total_paired_tasks": len(tasks),
    }


def get_query_for_condition(task: Task, condition: str) -> str:
    """
    Deterministic query selector per condition:
    - Condition A: original_query
    - Condition B: paraphrased_query
    - Condition C: paraphrased_query (strictly identical to Condition B)
    """
    cond = condition.upper()
    if cond == "CONDITION_A" or cond == "A":
        return task.original_query
    elif cond in ("CONDITION_B", "B", "CONDITION_C", "C"):
        return task.paraphrased_query
    else:
        raise ValueError(f"Unknown condition identifier: {condition}")
