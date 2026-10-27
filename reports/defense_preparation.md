# Faculty Defense & Presentation Guide
## AI465 Agentic AI Research Project

This guide provides speaking points, slide outlines, mathematical definitions, and likely faculty questions for defending your research project:
**"Joint Input Robustness of Budget-Constrained Tool Calling"**.

---

## 1. High-Level Elevator Pitch (60 Seconds)

> *"Good morning, professors. My research project for AI465 is titled **Joint Input Robustness of Budget-Constrained Tool Calling**, building on recent findings from Rabinovich & Anaby-Tavor (TrustNLP 2025).
>
> In production agentic AI systems, two realities collide: user queries are expressed naturally using paraphrasing rather than rigid syntax, and toolboxes contain dozens of tools with overlapping vocabulary and realistic distractors. Declaring all candidate tools in every prompt balloons token costs (~2,080 tokens per call) and increases distractor selection errors.
>
> To solve this, I designed a three-condition experiment comparing (A) Original Queries with Full Toolkits, (B) Paraphrased Queries with Full Toolkits, and (C) Paraphrased Queries with a Top-3 Semantic Shortlist retrieved using Sentence Transformers (`all-MiniLM-L6-v2`).
>
> Testing on Google Gemini (`gemini-flash-lite-latest`) across 30 tasks and 24 candidate tools, our findings demonstrate that **Sentence Transformer top-3 semantic shortlisting achieves an 85.5% reduction in prompt token consumption** while **preserving high function-calling accuracy** and achieving **100% retrieval recall**. The entire system is implemented in Python, CPU-friendly, tested with 23 unit tests, and fully reproducible."*

---

## 2. 8-Minute Presentation Slide Deck Outline

### Slide 1: Title & Motivation
- **Title**: Joint Input Robustness of Budget-Constrained Tool Calling
- **Problem**: Large toolboxes scale input token costs linearly ($O(M)$ where $M$ is tool count) and expose LLMs to semantic distractors.
- **Goal**: Quantify robustness under paraphrasing and test if dense semantic pruning preserves fidelity while minimizing token expense.

### Slide 2: Research Question & Hypotheses
- **RQ**: *How does the combination of paraphrased user requests and an expanded set of available tools affect an LLM's tool-calling accuracy and token cost? Can semantic tool shortlisting reduce token consumption while preserving accuracy?*
- **H1**: Paraphrasing increases distractor selection in large toolboxes.
- **H2**: Full 24-tool schema incurs heavy token overhead (>2,000 tokens/call).
- **H3**: Top-3 shortlisting cuts token overhead by 70–85% without sacrificing tool-calling precision.

### Slide 3: Related Work & Theoretical Gap
- **Ella Rabinovich & Ateret Anaby-Tavor (TrustNLP @ ACL 2025)**: Evaluated query perturbations on tool calling, highlighting sensitivity to phrasing.
- **BFCL (Patil et al., PMLR 2025)**: Standardized single-turn AST evaluation and distractor tool challenges.
- **Our Gap**: Integrating dense neural semantic tool pruning (`all-MiniLM-L6-v2`) into the robustness pipeline under budget constraints.

### Slide 4: Experimental Design (3 Conditions)
- **Condition A**: Original Query + 24 Tools (12 targets + 12 realistic distractors).
- **Condition B**: Meaning-Preserving Paraphrase + 24 Tools.
- **Condition C**: Meaning-Preserving Paraphrase + Top-3 Shortlist via Sentence Transformers (`all-MiniLM-L6-v2`).

### Slide 5: System Architecture & Implementation
- **Tool Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` dense embedding space (384 dimensions) with cosine similarity and in-memory embedding caching.
- **Model Client**: Native structured function calling via Google Gemini API with disk-backed MD5 caching, exponential backoff, and 3.0s pacing.

### Slide 6: Key Results
- **Token Reduction**: Condition C achieved **85.5% token savings** (dropping from ~2,080 to ~300 prompt tokens).
- **Accuracy**: Preserved tool-selection accuracy across evaluated conditions.
- **Retrieval Recall**: 100% top-3 recall with Sentence Transformers on the validated benchmark.

### Slide 7: Architectural Discovery (TF-IDF vs Sentence Transformers)
- **Lexical Failure**: Lexical matching (TF-IDF) struggles on paraphrased inputs (top-3 recall dropped to 66.7% when queries lack keyword overlap).
- **Dense Resolution**: Dense embeddings capture conceptual semantics, recovering 100% top-3 recall without needing exact keyword overlap.

### Slide 8: Conclusion & Future Scope
- **Conclusion**: Semantic tool shortlisting is an essential architectural pattern for cost-effective, robust agentic workflows.
- **Future Work**: Evaluating open-source LLMs (Llama-3), scaling to 100+ tools, and multi-turn dialogs.

---

## 3. Likely Faculty Questions & How to Answer Them

### Q1: *"Why did you use Sentence Transformers instead of TF-IDF or full BM25?"*
> **Answer:** *"TF-IDF and BM25 rely strictly on lexical keyword overlap. When users paraphrase queries—using synonyms, passive voice, or indirect phrasing (e.g. 'How much does it run me to transport this parcel?' instead of 'calculate shipping fee')—the vocabulary mismatch problem causes lexical retrievers to miss the ground truth tool. In our experiments, Sentence Transformers (`all-MiniLM-L6-v2`) maps queries and tool descriptions into a shared 384-dimensional dense semantic space, capturing conceptual intent regardless of surface wording and achieving 100% Top-3 Recall."*

### Q2: *"Is top-3 shortlisting too aggressive? What if the correct tool is ranked 4th?"*
> **Answer:** *"Top-3 represents an 87.5% reduction from our 24-tool library, maximizing token savings. However, in our architecture, the LLM's accuracy is upper-bounded by retrieval recall: $\text{Acc}_{\text{LLM}} \le \text{Recall}@K$. For production systems with hundreds of tools, a dynamic threshold (e.g. cosine similarity > 0.45) or a slightly wider window ($K=5$) could provide an optimal safety margin while still saving ~75% of prompt tokens."*

### Q3: *"How did you ensure the evaluation was fair and not hardcoded?"*
> **Answer:** *"All 30 tasks are evaluated across identical schemas with strictly paired task IDs. The retrieval ranking algorithm is strictly label-blind—it has zero access to the expected tool name or arguments at inference time. All predictions are logged incrementally to `results/predictions.csv`, verified by 23 automated unit tests, and can be reproduced by running `pytest` and `scripts/run_experiment.py`."*
