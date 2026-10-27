import json
from pathlib import Path

nb_path = Path("notebooks/Joint_Input_Robustness_Experiment.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 0: title and subtitle
nb["cells"][0]["source"] = [
    "# Joint Input Robustness of Budget-Constrained Tool Calling\n",
    "## AI465 Agentic AI Research Project\n",
    "**Primary Academic Reference:** *On the Robustness of Agentic Function Calling* — Ella Rabinovich & Ateret Anaby-Tavor (TrustNLP 2025)\n",
    "\n",
    "This notebook demonstrates and evaluates the **Joint Input Robustness** of an LLM agent under simultaneous **query paraphrasing** and **toolkit scaling**, comparing baseline full toolkits against **Sentence Transformer-based semantic tool shortlisting** (`all-MiniLM-L6-v2`).\n"
]

# Update Cell 16 & 17: Sentence Transformers Retrieval
nb["cells"][16]["source"] = [
    "## 11. Sentence Transformer Semantic Tool Retrieval (all-MiniLM-L6-v2)\n",
    "We employ `sentence-transformers/all-MiniLM-L6-v2` to map user queries and standardized tool definitions into a shared 384-dimensional dense semantic embedding space. In Condition C, we retrieve the Top-3 tools via cosine similarity, pruning ~85% of tool schema tokens before invoking the LLM.\n"
]

nb["cells"][17]["source"] = [
    "from src.tool_shortlisting import SentenceTransformerToolShortlister, TfidfToolShortlister\n",
    "\n",
    "# Initialize neural semantic shortlister (CPU-friendly, loaded once)\n",
    "st_shortlister = SentenceTransformerToolShortlister()\n",
    "test_task = tasks[0]\n",
    "res = st_shortlister.shortlist(\n",
    "    query=test_task.paraphrased_query,\n",
    "    tools=tools,\n",
    "    expected_tool_name=test_task.expected_tool_name,\n",
    "    top_k=3\n",
    ")\n",
    "\n",
    "print(f\"Original Query:     '{test_task.original_query}'\")\n",
    "print(f\"Paraphrased Query:  '{test_task.paraphrased_query}'\")\n",
    "print(f\"Expected Tool:      '{test_task.expected_tool_name}'\")\n",
    "print(f\"Retrieved Top-3:    {res.tool_names}\")\n",
    "print(f\"Similarity Scores:  {res.scores}\")\n",
    "print(f\"Target in Top-3:    {res.expected_tool_present}\")\n",
    "print(f\"Retrieval Latency:  {res.retrieval_latency_ms:.2f} ms\")\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully updated with Sentence Transformers retrieval.")
