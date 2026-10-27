# Experimental Results and Artifacts

This directory stores all experimental outputs, predictions, metadata, and computed summary metrics for the research project:
**"Joint Input Robustness of Budget-Constrained Tool Calling"** (AI465 Agentic AI Research Project).

## Files in this Directory

1. `predictions.csv`:
   - Comprehensive record of every attempted API call.
   - Includes full metadata: `task_id`, `condition`, `model_name`, `query_text`, `candidate_tools_count`, `predicted_tool`, `predicted_arguments`, `expected_tool`, `expected_arguments`, `tool_name_correct`, `arguments_correct`, `exact_call_correct`, `shortlist_recall`, `shortlist_tools`, `api_success`, `input_token_count`, `output_token_count`, `token_count_source`, `latency_seconds`, `is_cached`, `is_dry_run`, `error_message`, and `timestamp`.
   - Contains 20 attempted evaluations (19 successful authentic calls before the Google AI Studio 20 requests/day free-tier quota ceiling was reached).

2. `experiment_results.csv`:
   - The strictly paired subset of 15 evaluations across Condition A, Condition B, and Condition C evaluated on identical benchmark tasks (`task_01` through `task_05`).
   - Serves as the ground data for paired statistical testing, token comparisons, and publication figures.

3. `summary_metrics.json`:
   - Aggregated metrics across conditions: tool selection accuracy, exact match accuracy, argument accuracy, mean/median input tokens, token savings percentage (85.58%), shortlist recall (100%), and paired difference statistics with 95% bootstrap confidence intervals.

4. `experiment_metadata.json`:
   - Run configuration: model name (`gemini-3.5-flash`), masked API key, random seed (42), total candidate tools (20), shortlist method (`tfidf_top3`), and timestamp.

5. `cache/`:
   - Disk-backed JSON responses keyed by MD5 hashes of `(model, query, candidate_tools)`.
   - Prevents duplicate API calls, preserves budget, and enables instant local reproducibility.

## Quota and Academic Integrity Note
- Evaluated on Google AI Studio Free Tier endpoint (`gemini-3.5-flash`).
- When the daily 20-request free-tier ceiling (`generatelanguage.googleapis.com/generate_content_free_tier_requests`, limit: 20/day) was encountered, the experiment stopped gracefully.
- All numbers in these files reflect genuine API calls without synthetic fabrication.
