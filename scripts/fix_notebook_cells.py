import json
from pathlib import Path

nb_path = Path("notebooks/Joint_Input_Robustness_Experiment.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Code Cell 14, 15, 16, 17
code_indices = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]

# Cell 14 (index code_indices[13])
nb["cells"][code_indices[13]]["source"] = [
    "from src.evaluation import compute_condition_metrics\n",
    "from src.token_accounting import compute_token_savings\n",
    "\n",
    "# Compute condition metrics from task-level predictions\n",
    "metrics_a = compute_condition_metrics(df_all[df_all['condition'] == 'Condition_A'])\n",
    "metrics_b = compute_condition_metrics(df_all[df_all['condition'] == 'Condition_B'])\n",
    "metrics_c = compute_condition_metrics(df_all[df_all['condition'] == 'Condition_C'])\n",
    "\n",
    "savings = compute_token_savings(metrics_b['mean_input_tokens'], metrics_c['mean_input_tokens'])\n",
    "\n",
    "print('--- 30-TASK BENCHMARK EVALUATION SUMMARY ---')\n",
    "print(f\"Condition A Exact Accuracy: {metrics_a['exact_call_accuracy']*100:.1f}%, Mean Tokens: {metrics_a['mean_input_tokens']:.1f}\")\n",
    "print(f\"Condition B Exact Accuracy: {metrics_b['exact_call_accuracy']*100:.1f}%, Mean Tokens: {metrics_b['mean_input_tokens']:.1f}\")\n",
    "print(f\"Condition C Exact Accuracy: {metrics_c['exact_call_accuracy']*100:.1f}%, Mean Tokens: {metrics_c['mean_input_tokens']:.1f}\")\n",
    "print(f\"Condition C Token Savings:  {savings:.2f}%\")\n"
]

# Cell 15 (index code_indices[14])
nb["cells"][code_indices[14]]["source"] = [
    "from src.evaluation import compute_paired_comparison\n",
    "\n",
    "p_ab = compute_paired_comparison(df_all, 'Condition_A', 'Condition_B')\n",
    "p_bc = compute_paired_comparison(df_all, 'Condition_B', 'Condition_C')\n",
    "\n",
    "print(f\"Condition A vs B Paired Diff: {p_ab['mean_difference']:.4f}, 95% CI: {p_ab['bootstrap_ci_95']}, McNemar p: {p_ab['mcnemar_p_value']}\")\n",
    "print(f\"Condition B vs C Paired Diff: {p_bc['mean_difference']:.4f}, 95% CI: {p_bc['bootstrap_ci_95']}, McNemar p: {p_bc['mcnemar_p_value']}\")\n"
]

# Cell 16 (index code_indices[15])
nb["cells"][code_indices[15]]["source"] = [
    "from src.visualization import generate_all_figures\n",
    "from IPython.display import Image, display\n",
    "\n",
    "generate_all_figures(df_all)\n",
    "\n",
    "fig_names = [\n",
    "    'accuracy_comparison.png',\n",
    "    'token_comparison.png',\n",
    "    'shortlist_recall.png',\n",
    "    'task_level_comparison.png',\n",
    "    'failure_breakdown.png'\n",
    "]\n",
    "\n",
    "for name in fig_names:\n",
    "    p = PROJECT_ROOT / 'figures' / name\n",
    "    if p.exists():\n",
    "        print(f\"Figure: {name}\")\n",
    "        display(Image(filename=str(p), width=650))\n"
]

# Cell 17 (index code_indices[16])
nb["cells"][code_indices[16]]["source"] = [
    "rec_c = df_all[df_all['condition'] == 'Condition_C']['shortlist_recall'].mean() * 100\n",
    "\n",
    "summary_table = pd.DataFrame([\n",
    "    {\n",
    "        'Condition': 'Condition A (Original + Full 24)',\n",
    "        'Tasks': metrics_a['evaluable_tasks'],\n",
    "        'Tool Selection Acc': f\"{metrics_a['tool_selection_accuracy']*100:.1f}%\",\n",
    "        'Exact Call Acc': f\"{metrics_a['exact_call_accuracy']*100:.1f}%\",\n",
    "        'Mean Input Tokens': f\"{metrics_a['mean_input_tokens']:.1f}\",\n",
    "        'Shortlist Recall@3': 'N/A'\n",
    "    },\n",
    "    {\n",
    "        'Condition': 'Condition B (Paraphrased + Full 24)',\n",
    "        'Tasks': metrics_b['evaluable_tasks'],\n",
    "        'Tool Selection Acc': f\"{metrics_b['tool_selection_accuracy']*100:.1f}%\",\n",
    "        'Exact Call Acc': f\"{metrics_b['exact_call_accuracy']*100:.1f}%\",\n",
    "        'Mean Input Tokens': f\"{metrics_b['mean_input_tokens']:.1f}\",\n",
    "        'Shortlist Recall@3': 'N/A'\n",
    "    },\n",
    "    {\n",
    "        'Condition': 'Condition C (Paraphrased + Top-3 Shortlist)',\n",
    "        'Tasks': metrics_c['evaluable_tasks'],\n",
    "        'Tool Selection Acc': f\"{metrics_c['tool_selection_accuracy']*100:.1f}%\",\n",
    "        'Exact Call Acc': f\"{metrics_c['exact_call_accuracy']*100:.1f}%\",\n",
    "        'Mean Input Tokens': f\"{metrics_c['mean_input_tokens']:.1f}\",\n",
    "        'Shortlist Recall@3': f\"{rec_c:.1f}%\"\n",
    "    }\n",
    "])\n",
    "print(summary_table.to_string(index=False))\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook cells 14-17 fixed.")
