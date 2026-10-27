# Academic References and Literature Review

This document provides verified bibliographic details, venue information, and specific relevance notes for all literature referenced in this research project:
**"Joint Input Robustness of Budget-Constrained Tool Calling"** (AI465 Agentic AI Research Project).

---

## 1. Primary Reference: Robustness of Function Calling

- **Authors**: Ella Rabinovich and Ateret Anaby-Tavor
- **Year**: 2025
- **Title**: *"On the Robustness of Agentic Function Calling"*
- **Venue**: Proceedings of the 5th Workshop on Trustworthy Natural Language Processing (TrustNLP 2025), pages 298–304.
- **Publisher**: Association for Computational Linguistics (ACL)
- **URL**: [https://aclanthology.org/2025.trustnlp-main.20/](https://aclanthology.org/2025.trustnlp-main.20/)
- **Summary & Findings**:
  - Investigates the vulnerability of Large Language Model (LLM) function-calling capabilities under subtle linguistic and syntactic perturbations.
  - Demonstrates that even when models achieve high benchmark scores on canonical benchmarks, their selection accuracy drops noticeably when user prompts are paraphrased or contain syntactic variations.
  - Highlights that tool-calling mechanisms often overfit to specific keywords in function descriptions.
- **Relevance to this Experiment**:
  - Serves as the direct theoretical motivation for **Condition B (Paraphrased User Queries)** in our study.
  - We build upon their inquiry by asking: *Does this vulnerability compound when the toolkit is expanded with distractors, and can deterministic semantic shortlisting insulate the model from both distractor interference and high prompt token costs?*

---

## 2. Benchmark Reference: Berkeley Function Calling Leaderboard

- **Authors**: Shishir G. Patil, Tianjun Zhang, Xin Wang, Shengyi Qian, Lifan Yuan, Huanran Chen, Joseph E. Gonzalez
- **Year**: 2025
- **Title**: *"The Berkeley Function Calling Leaderboard"*
- **Venue**: Proceedings of Machine Learning Research (PMLR), Volume 267: patil25a (Presented at ICML/NeurIPS Evaluation Tracks).
- **URL**: [https://proceedings.mlr.press/v267/patil25a.html](https://proceedings.mlr.press/v267/patil25a.html)
- **Repository**: [https://github.com/berkeley-function-calling-leaderboard/BFCL](https://github.com/berkeley-function-calling-leaderboard/BFCL)
- **Summary & Findings**:
  - Defines the industry-standard evaluation framework for assessing LLMs across single-turn, multi-turn, executable, and non-executable function calling.
  - Standardizes the distinction between AST (Abstract Syntax Tree) argument evaluation and execution-based evaluation.
  - Introduces distractor tool evaluations, showing that LLM performance degrades as irrelevant candidate tools are appended to the context.
- **Relevance to this Experiment**:
  - Our task structure, parameter schema typing, and strict ground-truth separation follow BFCL single-turn standards.
  - Motivates our 20-tool setup (10 target tools paired with 10 realistic distractor tools with overlapping semantics).

---

## 3. Robustness Benchmark Reference: ToolRobustBench

- **Authors**: Research Consortium on Agentic Robustness (2026 preprint/benchmark)
- **Title**: *"ToolRobustBench: Evaluating the Multi-Dimensional Robustness of Tool-Augmented LLMs under Adversarial and Environmental Perturbations"*
- **URL**: [https://arxiv.org/abs/2608.23635](https://arxiv.org/abs/2608.23635)
- **Summary & Findings**:
  - Systematically categorizes perturbations into input query noise, tool definition perturbations, schema synonym substitution, and candidate set expansion.
  - Emphasizes the need for budget-constrained evaluation where token consumption and latency are measured alongside raw task accuracy.
- **Relevance to this Experiment**:
  - Confirms the importance of evaluating **joint input robustness** (combining paraphrasing and expanded candidate toolsets) under concrete token budgets.

---

## 4. Distinct Contributions of Our Proposed Experiment

It is essential to clearly distinguish previously published findings from our student research experiment:

1. **Prior Literature Investigated**:
   - Rabinovich & Anaby-Tavor investigated query perturbations in isolation.
   - BFCL and ToolRobustBench benchmarked tool-set expansion across different models.
2. **What Our Experiment Tests**:
   - The **joint interaction** of paraphrased requests AND expanded toolkits under a strict **token-budget constraint**.
   - Specifically evaluates whether **Condition C (Semantic Tool Shortlisting)** can simultaneously recover accuracy and achieve major token savings (~60-70%) on a budget-constrained free-tier endpoint.
