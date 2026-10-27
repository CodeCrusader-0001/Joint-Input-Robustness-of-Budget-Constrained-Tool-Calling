# Joint Input Robustness of Budget-Constrained Tool Calling

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Course](https://img.shields.io/badge/Course-AI465%20Agentic%20AI%20Research%20Project-green.svg)]()
[![Embedding Model](https://img.shields.io/badge/Retriever-all--MiniLM--L6--v2-purple.svg)]()
[![Platform](https://img.shields.io/badge/Platform-CPU%20Friendly%20(No%20GPU%20Required)-orange.svg)]()

> **Final Laboratory Research Project for AI465 Agentic AI Course (B.Tech Artificial Intelligence)**  
> **Author:** Ashish Kumar  
> **Primary Academic Reference:** *On the Robustness of Agentic Function Calling* — Ella Rabinovich and Ateret Anaby-Tavor, TrustNLP 2025  

---

## 1. Executive Summary

This repository contains the complete implementation, datasets, retrieval algorithms, evaluation suite, and experimental results for the research paper:

> **"Joint Input Robustness of Budget-Constrained Tool Calling"**

### The Core Problem
In production agentic architectures, two real-world operational challenges collide:
1. **Linguistic Variance (Paraphrasing)**: User requests do not follow rigid prompt syntax; they feature indirect phrasing, synonyms, polite requests, and colloquial variations.
2. **Toolkit Scaling with Overlapping Semantics**: Toolboxes contain dozens of tools with overlapping vocabularies and adjacent functionality (e.g., `get_weather` vs. `get_forecast`, `calculate_shipping` vs. `track_shipment`).

Passing complete tool definition schemas (20–50 tools) into every LLM prompt balloons input token usage to 2,000+ tokens per query and increases distractor selection errors.

### The Proposed Method
We design and evaluate a **two-stage cascaded tool-calling pipeline**:
1. **Dense Semantic Retrieval (Stage 1)**: Using `sentence-transformers/all-MiniLM-L6-v2`, we encode user queries and candidate tool descriptions into a shared 384-dimensional dense semantic space and rank tools by cosine similarity.
2. **Constrained Function Calling (Stage 2)**: Only the Top-3 shortlisted tools are serialized into the LLM prompt, cutting token overhead by **~85%** while pruning distractors.

---

## 2. Experimental Conditions

The experiment compares **three strictly paired primary conditions** across identical benchmark tasks:

| Experimental Condition | Query Input Phrasing | Candidate Tools Exposed to LLM | Selection / Retrieval Mechanism | Primary Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Condition A (Baseline)** | Original canonical query | Complete 24-tool toolkit | Zero-shot native LLM function call | Upper-bound baseline performance under ideal phrasing. |
| **Condition B (Perturbation)** | Meaning-preserving paraphrase | Complete 24-tool toolkit | Zero-shot native LLM function call | Measure robustness degradation from linguistic variance alone. |
| **Condition C (Cascaded Shortlist)** | Meaning-preserving paraphrase (strictly identical to B) | Top-3 tools retrieved via Sentence Transformers | Two-stage: Dense Retrieval + LLM Tool Call | Determine if dense semantic retrieval reduces tokens while preserving accuracy. |

---

## 3. Semantically Similar Toolkit Architecture

The evaluation toolkit contains **24 realistic tools** divided into 12 target tools and 12 confusable distractors across 8 functional domains:
- **Documents**: `search_documents(query)` vs `search_document_by_title(title)` vs `search_document_by_keyword(keyword)`
- **E-Commerce**: `search_products(query)` vs `search_products_by_category(category)` vs `find_product_by_name(product_name)` vs `check_inventory(product_id)`
- **CRM**: `search_customers(query)` vs `get_customer_profile(customer_id)`
- **Orders**: `get_order_status(order_id)` vs `get_order_details(order_id)` vs `search_orders(query)`
- **Weather**: `get_weather(city, units)` vs `get_forecast(location, days)`
- **Dining**: `search_restaurants(cuisine, location)` vs `find_restaurants_nearby(location)`
- **Math & Finance**: `calculate(expression)` vs `convert_currency(amount, from_curr, to_curr)`
- **Logistics**: `calculate_shipping(weight, distance)` vs `track_shipment(tracking_number)`
- **Web & News**: `search_web(query)` vs `search_news(query)`
- **Calendar**: `get_calendar_events(date)` vs `schedule_meeting(title, start_time, attendees)`

All tools feature typed parameters, docstrings, and complete JSON schemas stored in `data/tool_definitions.json`.

---

## 4. Sentence Transformer Retrieval Pipeline

Retrieval in Condition C is performed using `sentence-transformers/all-MiniLM-L6-v2`:
1. **Document Representation**:
   ```text
   Name: {tool.name}
   Description: {tool.description}
   Parameters: {properties_str}
   ```
2. **Dense Embeddings & Caching**: Tool representations are embedded into 384-dimensional $\ell_2$-normalized vectors and cached in memory. Invalidation is tracked via an MD5 hash of tool schemas.
3. **Cosine Similarity**: Fast dot product scoring between normalized query vector and tool vectors:
   $$\text{Sim}(q, t_i) = \mathbf{e}_q \cdot \mathbf{e}_{t_i}$$
4. **Deterministic Ranking**: Ranked descending by similarity score, with alphabetical tie-breaking by tool name.
5. **Zero Label Leakage**: Retrieval is completely blind to ground-truth labels (`expected_tool_name` is never seen during ranking).

---

## 5. Evaluation Metrics & Formulation

1. **Tool-Selection Accuracy ($\text{Acc}_{\text{tool}}$)**:
   $$\text{Acc}_{\text{tool}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{t}_i = t_i^*)$$
2. **Exact Function-Call Accuracy ($\text{Acc}_{\text{exact}}$)**:
   $$\text{Acc}_{\text{exact}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{t}_i = t_i^* \land \hat{\mathbf{a}}_i \equiv \mathbf{a}_i^*)$$
3. **Provider-Reported Input Token Count**: Direct token count reported by Gemini API's `usageMetadata.promptTokenCount`.
4. **Input Token Reduction ($\Delta \text{Tokens}_{\%}$)**:
   $$\Delta \text{Tokens}_{\%} = \frac{\bar{T}_{\text{in}}^{(B)} - \bar{T}_{\text{in}}^{(C)}}{\bar{T}_{\text{in}}^{(B)}} \times 100\%$$
5. **Top-3 Retrieval Recall**:
   $$\text{Recall}@3 = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(t_i^* \in \text{Top-3}(q))$$
6. **Statistical Tests**: 95% empirical bootstrap confidence intervals (2,000 resamples) and two-sided exact McNemar tests on paired discordant tasks.

---

## 6. Directory Structure

```text
Joint_Input_Robustness/
├── data/
│   ├── tasks.json                 # 30 validated benchmark tasks with meaning-preserving paraphrases
│   ├── tool_definitions.json      # 24 OpenAPI tool definitions with distractor counterparts
│   └── README.md                  # Dataset provenance and annotation documentation
├── figures/                       # Publication-grade figures (300 DPI PNGs)
│   ├── accuracy_comparison.png    # Tool and exact call accuracy across A, B, C
│   ├── token_comparison.png       # Mean and median input token consumption
│   ├── shortlist_recall.png       # Top-3 shortlist recall rate
│   ├── failure_breakdown.png      # Condition C failure mode breakdown
│   └── task_level_comparison.png  # Paired task-by-task correctness matrix
├── notebooks/
│   └── Joint_Input_Robustness_Experiment.ipynb # Complete 24-section interactive research notebook
├── reports/
│   ├── methodology.md             # In-depth scientific methodology report
│   ├── defense_preparation.md     # Faculty viva defense speaking notes and Q&A guide
│   └── limitations.md             # Threats to validity and engineering constraints
├── results/
│   ├── predictions.csv            # Task-level experimental predictions across A, B, and C
│   ├── summary_metrics.json       # Computed aggregate metrics, token savings, and statistical tests
│   ├── experiment_metadata.json   # Model, timestamp, random seed, and execution environment metadata
│   └── historical_runs/           # Preserved baseline runs (TF-IDF vs Sentence Transformers)
├── scripts/
│   ├── run_experiment.py          # CLI runner for live or dry-run experiments
│   ├── execute_notebook.py        # Automated headless notebook executor
│   └── create_zip.py              # Packaging script for submission archive
├── src/
│   ├── config.py                  # Environment parameters and safety budgets
│   ├── dataset.py                 # Pydantic schemas, loaders, and mock tool registry
│   ├── evaluation.py              # Exact match metrics, bootstrap CI, and McNemar tests
│   ├── experiment.py              # 3-condition experiment execution runner
│   ├── function_calling.py        # Argument normalization and comparison logic
│   ├── model_client.py            # Gemini API client with caching and backoff
│   ├── paraphrasing.py            # Lexical diversity and Jaccard similarity metrics
│   ├── token_accounting.py       # Token usage tracking and savings computation
│   ├── tool_shortlisting.py       # Sentence Transformers & TF-IDF shortlisting engines
│   └── visualization.py           # Publication-grade Matplotlib figure generators
├── tests/                         # Pytest test suite (23 automated unit tests)
│   ├── test_dataset.py
│   ├── test_evaluation.py
│   ├── test_token_accounting.py
│   └── test_tool_shortlisting.py
├── .env.example                   # Environment variable template
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Locked dependencies
└── README.md                      # Project documentation
```

---

## 7. Quickstart & Reproduction Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Free Gemini API Key
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Edit `.env` and add your free Gemini API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
RATE_LIMIT_PACING_SECONDS=3.0
```

### Step 3: Run Unit Tests
```bash
pytest -v
```
All 23 unit tests run locally without GPU or network dependencies.

### Step 4: Execute Full Experiment
```bash
python scripts/run_experiment.py --retriever sentence_transformers
```
To run in mock dry-run mode (no API calls):
```bash
python scripts/run_experiment.py --dry-run
```

### Step 5: Run Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/Joint_Input_Robustness_Experiment.ipynb
```
Or execute headlessly:
```bash
python scripts/execute_notebook.py
```

---

## 8. Academic Integrity & Provenance

- **Original Paper Reference**: Rabinovich & Anaby-Tavor (*On the Robustness of Agentic Function Calling*, TrustNLP 2025).
- **Novel Contribution**: Introduction of Sentence Transformer (`all-MiniLM-L6-v2`) dense retrieval as a budget-constrained defense mechanism to prune prompt tokens while preserving accuracy.
- **Data Provenance**: Custom-curated 30-task pilot benchmark with verified meaning-preserving paraphrases and realistic distractor tools.
- **Empirical Execution**: Evaluated live on Google Gemini (`gemini-flash-lite-latest`) with recorded task-level predictions in `results/predictions.csv`.
