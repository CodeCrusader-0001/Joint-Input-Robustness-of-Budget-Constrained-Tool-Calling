# Threats to Validity and Experimental Limitations

Academic honesty and rigorous methodological transparency are central tenets of this research project. This document outlines known limitations, potential threats to validity, and caveats regarding the experimental results.

---

## 1. Threats to Internal Validity

1. **Deterministic vs. Stochastic Generation**:
   - The experiments were executed with `temperature = 0.0` to maximize reproducibility and minimize sampling noise.
   - While zero-temperature sampling reflects standard production deployment for structured tool calling, it may underestimate the variability observed when temperature > 0.
2. **Provider-Reported vs. Heuristic Tokens**:
   - For live API calls against Google Gemini (`gemini-flash-lite-latest`), official token usage metadata (`usageMetadata.promptTokenCount` and `candidatesTokenCount`) was parsed directly from API responses.
   - The system explicitly records the `token_count_source` for every row to prevent conflating estimated counts with provider counts.
3. **Pacing and Transient Rate Limits**:
   - The free-tier Gemini API enforces strict per-minute request quotas (~15 RPM). In conditions where consecutive requests encounter HTTP 429 or 503, exponential backoff (15s minimum delay) was enforced.

---

## 2. Threats to External Validity (Generalizability)

1. **Sample Size Constraints ($N=30$)**:
   - As an initial student research prototype designed to operate within free-tier API quotas and student laptop resource limits without a GPU, the benchmark consists of 30 curated single-turn tasks.
   - While 30 tasks are sufficient for paired hypothesis testing and statistical bootstrap estimation, the sample size produces wider confidence intervals than large-scale enterprise benchmarks (such as BFCL with 2,000+ tasks). Findings should be interpreted as indicative rather than globally conclusive.
2. **Single-Turn vs. Multi-Turn Function Calling**:
   - This experiment focuses strictly on single-turn function selection and argument extraction.
   - It does not evaluate multi-turn dialog state tracking, parallel tool execution, or agentic tool feedback loops where tool outputs are returned to the model across multiple reasoning turns.
3. **Single LLM Family**:
   - Evaluations were conducted on Google Gemini Flash Lite (`gemini-flash-lite-latest`). Different frontier models (e.g., GPT-4o, Claude 3.5 Sonnet, Llama-3-70B) may exhibit different baseline robustness curves against prompt paraphrasing or distractor tools.

---

## 3. Tool Retrieval Observations

1. **Dense Embeddings (`all-MiniLM-L6-v2`) vs. Lexical Retrieval (TF-IDF)**:
   - Dense semantic retrieval with `all-MiniLM-L6-v2` achieved **100% Top-3 Recall** on the validated benchmark, demonstrating superior semantic alignment compared to lexical n-gram TF-IDF on paraphrased queries with high lexical divergence.
   - Dense retrieval requires maintaining 384-dimensional floating point vectors in memory and incurs ~5–10 ms retrieval latency per query on CPU.
2. **Fixed Shortlist Cutoff ($K=3$)**:
   - Condition C employed a fixed $K=3$ cutoff (selecting 3 out of 24 candidate tools, an 87.5% pruning rate). A dynamic threshold (e.g., cosine similarity cutoff > 0.45) or variable $K \in \{3, 5, 8\}$ represents a valuable direction for future expansion.

---

## 4. Monetary vs. Token Cost Distinction

- All experiments were executed using Google AI Studio free-tier quotas.
- Consequently, monetary expenditure for this experiment was **$0.00**.
- We report actual input token counts and percentage token reductions rather than fabricating speculative dollar amounts.
