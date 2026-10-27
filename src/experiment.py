"""
Experiment orchestration engine for Joint Input Robustness.
Executes Condition A, Condition B, and Condition C in a strictly paired, resumable fashion.
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import (
    GEMINI_MODEL,
    RANDOM_SEED,
    RESULTS_DIR,
    get_masked_api_key,
    is_api_configured,
)
from src.dataset import Task, ToolDefinition, load_tasks, load_tools
from src.evaluation import compute_condition_metrics, compute_paired_comparison
from src.function_calling import compare_arguments
from src.model_client import ModelClient, ModelResponse
from src.paraphrasing import compute_lexical_diversity, get_query_for_condition
from src.token_accounting import compute_token_savings
from src.tool_shortlisting import (
    BaseToolShortlister,
    SentenceTransformerToolShortlister,
    TfidfToolShortlister,
    filter_tools_by_names,
)


class ExperimentRunner:
    """
    Orchestrates the 3-condition experiment across evaluation tasks.
    """
    def __init__(
        self,
        tasks: Optional[List[Task]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        client: Optional[ModelClient] = None,
        shortlister: Optional[BaseToolShortlister] = None,
        dry_run: bool = False,
        results_dir: Optional[Path] = None
    ):
        self.tools = tools or load_tools()
        self.tasks = tasks or load_tasks()
        self.results_dir = results_dir or RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_file = self.results_dir / "predictions.csv"
        self.summary_file = self.results_dir / "summary_metrics.json"
        self.metadata_file = self.results_dir / "experiment_metadata.json"

        self.client = client or ModelClient(dry_run=dry_run)
        self.shortlister = shortlister or SentenceTransformerToolShortlister()

    def _load_existing_predictions(self) -> pd.DataFrame:
        """Load already completed predictions to support resume functionality."""
        if self.predictions_file.exists():
            try:
                df = pd.read_csv(self.predictions_file)
                return df
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _save_predictions(self, rows: List[Dict[str, Any]]) -> None:
        """Append or update predictions CSV file."""
        if not rows:
            return
        new_df = pd.DataFrame(rows)
        if self.predictions_file.exists():
            try:
                existing_df = pd.read_csv(self.predictions_file)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                # Deduplicate on (task_id, condition) keeping the latest
                combined = combined.drop_duplicates(subset=["task_id", "condition"], keep="last")
                combined.to_csv(self.predictions_file, index=False)
                return
            except Exception:
                pass
        new_df.to_csv(self.predictions_file, index=False)

    def run(self, conditions: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run paired experiment across specified conditions:
        - Condition A: ORIGINAL + FULL TOOLKIT
        - Condition B: PARAPHRASED + FULL TOOLKIT
        - Condition C: PARAPHRASED + TOP-3 SHORTLIST
        """
        target_conditions = conditions or ["Condition_A", "Condition_B", "Condition_C"]
        existing_df = self._load_existing_predictions()

        completed_set = set()
        if not existing_df.empty and "task_id" in existing_df.columns and "condition" in existing_df.columns:
            for _, r in existing_df.iterrows():
                if r.get("api_success") is True or r.get("is_dry_run") is True:
                    completed_set.add((str(r["task_id"]), str(r["condition"])))

        # Reproducible task sequencing
        rng = random.Random(RANDOM_SEED)
        task_order = list(self.tasks)
        rng.shuffle(task_order)

        all_records = []
        start_time = datetime.now()

        for cond in target_conditions:
            for task in task_order:
                # Check if already completed in prior run
                if (task.task_id, cond) in completed_set:
                    continue

                query = get_query_for_condition(task, cond)
                shortlist_info = None

                # Determine candidate tool set for this condition
                if cond in ("Condition_A", "Condition_B"):
                    candidate_tools = self.tools
                    shortlist_tool_names = []
                    shortlist_recall = None
                else:  # Condition C: Paraphrased + Top-3 Shortlist
                    shortlist_info = self.shortlister.shortlist(
                        query=query,
                        tools=self.tools,
                        expected_tool_name=task.expected_tool_name,
                        top_k=3
                    )
                    shortlist_tool_names = shortlist_info.tool_names
                    shortlist_recall = 1 if shortlist_info.expected_tool_present else 0
                    candidate_tools = filter_tools_by_names(self.tools, shortlist_tool_names)

                # Call model client
                resp: ModelResponse = self.client.call_tool(
                    query=query,
                    tools=candidate_tools
                )

                # Evaluate correctness
                pred_tool = resp.predicted_tool
                pred_args = resp.predicted_arguments
                exp_tool = task.expected_tool_name
                exp_args = task.expected_arguments

                tool_correct = bool(pred_tool and (pred_tool == exp_tool))
                args_correct, arg_details = compare_arguments(pred_args, exp_args)
                exact_call_correct = bool(tool_correct and args_correct)

                record = {
                    "task_id": task.task_id,
                    "condition": cond,
                    "model_name": self.client.model_name,
                    "query_text": query,
                    "candidate_tools_count": len(candidate_tools),
                    "predicted_tool": pred_tool or "",
                    "predicted_arguments": json.dumps(pred_args),
                    "expected_tool": exp_tool,
                    "expected_arguments": json.dumps(exp_args),
                    "tool_name_correct": tool_correct,
                    "arguments_correct": args_correct,
                    "exact_call_correct": exact_call_correct,
                    "shortlist_recall": shortlist_recall if shortlist_recall is not None else -1,
                    "shortlist_tools": json.dumps(shortlist_tool_names),
                    "api_success": resp.api_success,
                    "input_token_count": resp.input_token_count,
                    "output_token_count": resp.output_token_count,
                    "token_count_source": resp.token_count_source,
                    "latency_seconds": resp.latency_seconds,
                    "is_cached": resp.is_cached,
                    "is_dry_run": resp.is_dry_run,
                    "error_message": resp.error_message or "",
                    "timestamp": datetime.now().isoformat()
                }

                all_records.append(record)
                # Incremental persist after each task
                self._save_predictions([record])
                completed_set.add((task.task_id, cond))

        # Reload complete combined predictions table
        final_df = pd.read_csv(self.predictions_file)
        self._generate_summary_reports(final_df, start_time)
        return final_df

    def _generate_summary_reports(self, df: pd.DataFrame, start_time: datetime) -> None:
        """Generate and save summary_metrics.json and experiment_metadata.json."""
        cond_a = compute_condition_metrics(df[df["condition"] == "Condition_A"])
        cond_b = compute_condition_metrics(df[df["condition"] == "Condition_B"])
        cond_c = compute_condition_metrics(df[df["condition"] == "Condition_C"])

        # Condition C shortlist recall
        df_c = df[df["condition"] == "Condition_C"]
        recall_valid = df_c[df_c["shortlist_recall"] >= 0]["shortlist_recall"]
        mean_recall = float(recall_valid.mean()) if len(recall_valid) > 0 else 0.0

        # Token savings: Condition C vs Condition B
        b_tok = cond_b.get("mean_input_tokens", 0.0)
        c_tok = cond_c.get("mean_input_tokens", 0.0)
        token_savings_pct = compute_token_savings(b_tok, c_tok)

        # Paired comparisons
        paired_a_vs_b = compute_paired_comparison(df, "Condition_A", "Condition_B")
        paired_b_vs_c = compute_paired_comparison(df, "Condition_B", "Condition_C")
        paired_a_vs_c = compute_paired_comparison(df, "Condition_A", "Condition_C")

        # Lexical diversity
        lexical_metrics = compute_lexical_diversity(self.tasks)

        summary = {
            "experiment_title": "Joint Input Robustness of Budget-Constrained Tool Calling",
            "model_evaluated": self.client.model_name,
            "total_tasks_evaluated": len(self.tasks),
            "is_dry_run": bool(df["is_dry_run"].all()),
            "condition_metrics": {
                "Condition_A_Original_Full": cond_a,
                "Condition_B_Paraphrased_Full": cond_b,
                "Condition_C_Paraphrased_Shortlist": cond_c
            },
            "shortlist_recall_top3": round(mean_recall, 4),
            "token_savings_condition_c_vs_b_pct": round(token_savings_pct, 2),
            "paired_statistical_comparisons": {
                "Condition_A_vs_Condition_B": paired_a_vs_b,
                "Condition_B_vs_Condition_C": paired_b_vs_c,
                "Condition_A_vs_Condition_C": paired_a_vs_c
            },
            "paraphrase_lexical_metrics": lexical_metrics,
            "total_evaluations_recorded": len(df)
        }

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        metadata = {
            "timestamp_start": start_time.isoformat(),
            "timestamp_end": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED,
            "model_name": self.client.model_name,
            "api_key_masked": get_masked_api_key(),
            "total_candidate_tools": len(self.tools),
            "total_target_tools": sum(1 for t in self.tools if not t.is_distractor),
            "total_distractor_tools": sum(1 for t in self.tools if t.is_distractor),
            "shortlist_method": getattr(self.shortlister, "method_name", "sentence_transformers"),
            "embedding_model": getattr(self.shortlister, "model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            "is_real_api_run": not bool(df["is_dry_run"].all())
        }

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
