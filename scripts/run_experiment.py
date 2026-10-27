"""
Command-line execution script for the research experiment.
Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --dry-run
    python scripts/run_experiment.py --task-limit 10
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GEMINI_MODEL, is_api_configured
from src.dataset import load_tasks, load_tools
from src.experiment import ExperimentRunner
from src.model_client import ModelClient
from src.visualization import generate_all_figures


def main():
    parser = argparse.ArgumentParser(
        description="Run Joint Input Robustness tool-calling experiment."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in mock dry-run mode without making actual API calls."
    )
    parser.add_argument(
        "--task-limit",
        type=int,
        default=0,
        help="Limit number of tasks to evaluate (0 means all tasks in dataset)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GEMINI_MODEL,
        help=f"Target Gemini model (default: {GEMINI_MODEL})."
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="sentence_transformers",
        choices=["sentence_transformers", "tfidf"],
        help="Retrieval algorithm for Condition C (default: sentence_transformers)."
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("JOINT INPUT ROBUSTNESS EXPERIMENT")
    print("AI465 Agentic AI Research Project")
    print("=" * 70)

    # API configuration check
    api_ready = is_api_configured()
    dry_run_mode = args.dry_run or (not api_ready)

    if not api_ready and not args.dry_run:
        print("[!] NOTICE: GEMINI_API_KEY is not configured or invalid.")
        print("    Running pipeline in demonstration / dry-run mode.")
    elif dry_run_mode:
        print("[*] Running in requested DRY-RUN mode.")
    else:
        print(f"[*] API Key detected. Live evaluation target model: {args.model}")

    # Load data
    tasks = load_tasks(limit=args.task_limit if args.task_limit > 0 else None)
    tools = load_tools()

    print(f"[*] Loaded {len(tasks)} tasks and {len(tools)} candidate tools.")
    print(f"[*] Condition C Retriever: {args.retriever}")
    print("    Conditions to evaluate:")
    print(f"      - Condition A: Original Query + Full {len(tools)} Tools")
    print(f"      - Condition B: Paraphrased Query + Full {len(tools)} Tools")
    print(f"      - Condition C: Paraphrased Query + Top-3 Shortlist ({args.retriever})")

    client = ModelClient(
        model_name=args.model,
        dry_run=dry_run_mode
    )

    from src.tool_shortlisting import SentenceTransformerToolShortlister, TfidfToolShortlister
    shortlister = (
        SentenceTransformerToolShortlister()
        if args.retriever == "sentence_transformers"
        else TfidfToolShortlister()
    )

    runner = ExperimentRunner(
        tasks=tasks,
        tools=tools,
        client=client,
        shortlister=shortlister,
        dry_run=dry_run_mode
    )

    print("\n[*] Starting evaluation...")
    df_results = runner.run()

    print(f"\n[+] Completed! Total evaluations recorded: {len(df_results)}")

    if not args.no_figures:
        print("[*] Generating publication figures in figures/ ...")
        generate_all_figures(df_results)
        print("[+] Figures successfully generated.")

    # Print summary results table
    print("\n" + "=" * 70)
    print("EXPERIMENTAL SUMMARY RESULTS TABLE")
    print("=" * 70)
    for cond in ["Condition_A", "Condition_B", "Condition_C"]:
        sub = df_results[df_results["condition"] == cond]
        tool_acc = sub["tool_name_correct"].mean() * 100
        exact_acc = sub["exact_call_correct"].mean() * 100
        mean_tok = sub["input_token_count"].mean()
        med_tok = sub["input_token_count"].median()
        succ_calls = sub["api_success"].sum()
        total_calls = len(sub)
        print(f"\n--- {cond} ---")
        print(f"  Evaluable Tasks:        {total_calls}")
        print(f"  Successful API Calls:   {succ_calls}/{total_calls}")
        print(f"  Tool Selection Acc:     {tool_acc:.1f}%")
        print(f"  Exact Call Acc:         {exact_acc:.1f}%")
        print(f"  Mean Input Tokens:      {mean_tok:.1f}")
        print(f"  Median Input Tokens:    {med_tok:.1f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
