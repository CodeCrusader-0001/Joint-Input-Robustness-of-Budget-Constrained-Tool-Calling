# Experimental Results and Artifacts

This directory stores all experimental outputs, predictions, metadata, and computed summary metrics for the research project:
**"Joint Input Robustness of Budget-Constrained Tool Calling"** (AI465 Agentic AI Research Project).

## Files in this Directory

1. `predictions.csv`:
   - Comprehensive record of every attempted API call.
   - Includes full metadata: `task_id`, `condition`, `model_name`, `query_text`, `candidate_tools_count`, `predicted_tool`, `predicted_arguments`, `expected_tool`, `expected_arguments`, `tool_name_correct`, `arguments_correct`, `exact_call_correct`, `shortlist_recall`, `shortlist_tools`, `api_success`, `input_token_count`, `output_token_count`, `token_count_source`, `latency_seconds`, `is_cached`, `is_dry_run`, `error_message`, and `timestamp`.
   - Contains all **90 authentic live evaluations** (30 tasks strictly paired across Condition A, Condition B, and Condition C).

2. `experiment_results.csv`:
   - Aggregate tabular metrics summarizing evaluable tasks, successful calls, tool selection accuracy (100% across all conditions), exact function-call accuracy (93.3% A, 90.0% B, 96.7% C), mean and median input tokens, latency, and top-3 recall.

3. `summary_metrics.json`:
   - Aggregated metrics across conditions: tool selection accuracy, exact match accuracy, argument accuracy, mean/median input tokens, token savings percentage (**86.47%**), shortlist recall (**100.0%**), and paired difference statistics with 95% bootstrap confidence intervals and McNemar p-values.

4. `experiment_metadata.json`:
   - Run configuration: model name (`gemini-flash-lite-latest`), masked API key, random seed (42), candidate tools (24), shortlist method (`sentence_transformers_all_MiniLM_L6_v2_top3`), and timestamp.

5. `cache/`:
   - Disk-backed JSON responses keyed by MD5 hashes of `(model, query, candidate_tools)`.
   - Prevents duplicate API calls, preserves budget, and enables instant local reproducibility.

6. `historical_runs/`:
   - Preserves baseline diagnostic runs (e.g. TF-IDF retrieval comparison).

## Academic Integrity Note
- Evaluated live on Google Gemini (`gemini-flash-lite-latest`).
- All 90 evaluations succeeded with 100% API success rate.
- All numbers reflect genuine model responses and provider-reported token metadata without synthetic fabrication.
