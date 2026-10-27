# Dataset and Tool Definitions Provenance

This directory contains the standardized evaluation dataset and tool schemas used for the research experiment:
**"Joint Input Robustness of Budget-Constrained Tool Calling"** (AI465 Agentic AI Research Project).

## Files Overview

1. `tasks.json`:
   - 30 meticulously constructed and validated evaluation tasks across 10 functional domains (search, orders, products, CRM, weather, dining, calculation, logistics, web search, calendar).
   - Every task includes:
     - `task_id`: Unique identifier (`task_01` to `task_30`).
     - `original_query`: Direct canonical user request.
     - `paraphrased_query`: Verified meaning-preserving paraphrase retaining all entities, constraints, and intent.
     - `tool_definitions`: Candidate tool names.
     - `expected_tool_name` / `expected_tool`: Ground-truth target function.
     - `expected_arguments`: Ground-truth parameter dictionary.
     - `task_category`: Domain categorization.
     - `difficulty_category`: Pilot difficulty rating (easy, medium, hard).
     - `data_source`: Custom curated academic benchmark.
     - `validation_status`: `human_verified`.
   - Conforms to Berkeley Function Calling Leaderboard (BFCL) single-turn conventions.

2. `tool_definitions.json`:
   - Comprehensive 24-tool toolkit consisting of 12 target tools and 12 realistic distractor tools with overlapping vocabulary and similar functional descriptions.
   - Compliant with OpenAPI / JSON Schema format accepted by Gemini and OpenAI function-calling APIs.
   - Contains safe local mock implementations for deterministic testing and validation.

## Provenance and License

- **Source**: Curated academic evaluation suite authored specifically for the AI465 Agentic AI course research project.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Validation**: All 30 tasks verified via unit tests ensuring 100% schema compliance, argument type validity, and ground-truth consistency.
