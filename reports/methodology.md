# Research Methodology & Experimental Protocol

**Project Title:** Joint Input Robustness of Budget-Constrained Tool Calling  
**Course:** AI465 Agentic AI Research Project  
**Author / Investigator:** Ashish Vishwakarma  
**Primary Reference:** *On the Robustness of Agentic Function Calling* — Ella Rabinovich and Ateret Anaby-Tavor, TrustNLP 2025  

---

## 1. Academic Background & Research Problem

In production Agentic AI workflows, large language models (LLMs) are augmented with specialized toolboxes to execute external actions (e.g., database lookups, financial transactions, calendar management, and calculation). 

The foundational study by **Rabinovich & Anaby-Tavor (TrustNLP 2025)** revealed a critical vulnerability in current LLM agents:
1. **Linguistic Sensitivity**: Minor phrasing perturbations, colloquial syntax, and paraphrasing can degrade function-calling accuracy by double-digit percentages.
2. **Distractor Vulnerability**: As candidate toolboxes scale beyond 10–20 tools, semantically adjacent tools (distractors) frequently mislead the LLM into calling incorrect or sub-optimal endpoints.

### What This Research Adds
While Rabinovich & Anaby-Tavor established the vulnerability of tool-calling LLMs under linguistic perturbation, they evaluated models under full candidate schemas without budgetary token constraints. In production, exposing 20–50 full tool schemas in every prompt:
- Inflates prompt token consumption by 500%–1,000% (~2,000 tokens per call).
- Increases latency and monetary inference cost.
- Exacerbates context clutter and distractor confusion.

This project proposes and investigates a **two-stage cascaded architecture**:
1. **Stage 1 (Neural Semantic Retrieval)**: A lightweight, CPU-efficient sentence-transformer model (`sentence-transformers/all-MiniLM-L6-v2`) encodes the user query and pre-filters the toolbox to a Top-3 shortlist via cosine similarity.
2. **Stage 2 (Agent Execution)**: The LLM receives *only* the top-3 shortlisted tools, drastically pruning distractors and cutting token usage.

### Core Research Questions
1. **RQ1**: How does query paraphrasing affect an LLM's function-calling accuracy when candidate toolsets contain semantically similar alternatives?
2. **RQ2**: Can dense semantic tool shortlisting using Sentence Transformers preserve or recover tool-calling accuracy while achieving substantial (>70%) input token reduction?
3. **RQ3**: How does neural semantic retrieval (`all-MiniLM-L6-v2`) compare against classical lexical retrieval (TF-IDF) under heavy query paraphrase?

---

## 2. Experimental Design (The Three Conditions)

To isolate the individual and joint effects of query variation and toolkit pruning, the experiment evaluates exactly **three primary experimental conditions** across identical paired tasks:

| Dimension | Condition A (Baseline) | Condition B (Paraphrased Full) | Condition C (Paraphrased Shortlist) |
| :--- | :--- | :--- | :--- |
| **User Query** | Original, direct phrasing | Meaning-preserving paraphrase | Meaning-preserving paraphrase (strictly identical to B) |
| **Available Tools** | Complete 24-tool toolkit | Complete 24-tool toolkit | Top-3 tools retrieved via Sentence Transformers |
| **Prompt Complexity** | Full JSON Schema (~2,080 tokens) | Full JSON Schema (~2,080 tokens) | Shortlisted JSON Schema (~300 tokens) |
| **Primary Goal** | Establish upper-bound baseline | Quantify degradation from paraphrase | Measure token savings and accuracy preservation |

All three conditions use identical models (`gemini-flash-lite-latest`), identical generation settings (temperature = 0.0), identical prompt structures, and identical ground-truth validation rules.

---

## 3. Semantically Similar Toolkit Architecture

To simulate real-world production systems where tool names and descriptions overlap, we constructed a toolkit of **24 tools** organized into **12 target tools** and **12 semantically similar distractor tools**:

| Domain | Target Tool | Confusable Distractor Tool | Distinct Key Functionality |
| :--- | :--- | :--- | :--- |
| **Documents** | `search_documents(query)` | `search_document_by_title(title)` | Full-text knowledge search vs exact title matching |
| **Documents** | `search_documents(query)` | `search_document_by_keyword(keyword)` | Free-text search vs taxonomy tag filtering |
| **E-Commerce** | `search_products(query)` | `search_products_by_category(category)` | Keyword search vs department browsing |
| **E-Commerce** | `search_products(query)` | `find_product_by_name(product_name)` | Broad item search vs exact brand name match |
| **E-Commerce** | `search_products(query)` | `check_inventory(product_id)` | Catalog discovery vs warehouse stock inspection |
| **CRM** | `search_customers(query)` | `get_customer_profile(customer_id)` | Multi-field search vs single account ID retrieval |
| **Orders** | `get_order_status(order_id)` | `get_order_details(order_id)` | Delivery progress status vs itemized billing breakdown |
| **Orders** | `get_order_status(order_id)` | `search_orders(query)` | Single order status vs multi-order search query |
| **Weather** | `get_weather(city, units)` | `get_forecast(location, days)` | Current atmospheric state vs multi-day future projection |
| **Dining** | `search_restaurants(cuisine, location)` | `find_restaurants_nearby(location)` | Cuisine-based search vs geographic proximity lookup |
| **Calculation** | `calculate(expression)` | `convert_currency(amount, from, to)` | Math expression evaluation vs foreign exchange conversion |
| **Logistics** | `calculate_shipping(weight, distance)` | `track_shipment(tracking_number)` | Freight fee estimation vs carrier transit checkpoint tracking |
| **Web Search** | `search_web(query)` | `search_news(query)` | Open internet search vs journalism/news headline search |
| **Calendar** | `get_calendar_events(date)` | `schedule_meeting(title, start_time)` | Schedule inspection vs creating a new calendar booking |

Each tool is defined with complete OpenAPI/Gemini parameter specifications, parameter types, descriptions, and required constraints.

---

## 4. Sentence Transformer Retrieval Algorithm

In Condition C, tools are ranked before prompting the LLM using `sentence-transformers/all-MiniLM-L6-v2`:

### 1. Document Representation
Each tool is formatted into a standardized textual representation:
```text
Name: get_order_status
Description: Retrieve the current delivery, fulfillment, or shipping progress status for a specific order ID.
Parameters: order_id, string.
```

### 2. Dense Vector Encoding
Both the user query $q$ and all tool representations $t_i \in \mathcal{T}$ are mapped into a dense 384-dimensional embedding space:
$$\mathbf{e}_q = \text{Encoder}(q), \quad \mathbf{e}_{t_i} = \text{Encoder}(t_i)$$
Embeddings are $\ell_2$-normalized so that cosine similarity reduces to an efficient inner product:
$$\text{Sim}(q, t_i) = \langle \mathbf{e}_q, \mathbf{e}_{t_i} \rangle$$

### 3. In-Memory & On-Disk Caching
Tool embeddings are computed once and cached in memory. An MD5 hash of the concatenated tool representations ensures the cache is automatically invalidated and recomputed if tool schemas change.

### 4. Deterministic Ranking & Tie-Breaking
Candidate tools are sorted descending by similarity score, with deterministic alphabetical tie-breaking by tool name:
$$\text{Rank}(t_i) = (-\text{Sim}(q, t_i), t_i.\text{name})$$
The Top-3 tools are extracted and provided to the LLM in Condition C.

### 5. Label-Blind Retrieval Guarantee
The retrieval algorithm operates strictly on the inference-time query text and candidate tool schemas. It has zero access to `expected_tool_name` or ground-truth arguments.

---

## 5. Evaluation Metrics

1. **Tool-Selection Accuracy ($\text{Acc}_{\text{tool}}$)**:
   $$\text{Acc}_{\text{tool}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{t}_i = t_i^*)$$
2. **Exact Function-Call Accuracy ($\text{Acc}_{\text{exact}}$)**:
   $$\text{Acc}_{\text{exact}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{t}_i = t_i^* \land \hat{\mathbf{a}}_i \equiv \mathbf{a}_i^*)$$
   where argument comparison accounts for type normalization (e.g., float vs int for numeric quantities) and case-insensitivity where appropriate.
3. **Provider-Reported Input Token Usage**:
   Direct token count recorded from the `usageMetadata.promptTokenCount` returned by the Gemini API.
4. **Token Reduction ($\Delta \text{Tokens}_{\%}$)**:
   $$\Delta \text{Tokens}_{\%} = \frac{\bar{T}_{\text{in}}^{(B)} - \bar{T}_{\text{in}}^{(C)}}{\bar{T}_{\text{in}}^{(B)}} \times 100\%$$
5. **Top-3 Retrieval Recall**:
   $$\text{Recall}@3 = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(t_i^* \in \mathcal{S}_i)$$
   Measures whether the ground-truth tool is successfully captured in the shortlist.
6. **Statistical Significance & Uncertainty**:
   - 95% empirical bootstrap confidence intervals (2,000 resamples) for paired accuracy differences.
   - Two-sided exact McNemar test on discordant task pairs between conditions.

---

## 6. Reproducibility & Execution Protocol

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PyTorch 2.0+ (CPU-friendly, no GPU required)
- `sentence-transformers >= 3.0`
- `google-genai` / `requests` / `pydantic` / `pandas` / `matplotlib` / `pytest`

### Environment Configuration
1. Clone the repository and navigate to `Joint_Input_Robustness`.
2. Copy `.env.example` to `.env` and set your free Gemini API key:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-flash-lite-latest
   RATE_LIMIT_PACING_SECONDS=3.0
   ```
3. Run the automated test suite:
   ```bash
   pytest -v
   ```
4. Execute the complete experiment:
   ```bash
   python scripts/run_experiment.py --retriever sentence_transformers
   ```
5. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/Joint_Input_Robustness_Experiment.ipynb
   ```
