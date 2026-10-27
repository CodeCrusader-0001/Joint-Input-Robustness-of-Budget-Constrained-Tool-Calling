"""
Unit tests for Sentence Transformers and TF-IDF tool shortlisting,
embedding caching, deterministic ranking, and label-blind retrieval.
"""

import numpy as np
import pytest
from src.dataset import ToolDefinition, ToolParameters, load_tasks, load_tools
from src.tool_shortlisting import (
    SentenceTransformerToolShortlister,
    TfidfToolShortlister,
    filter_tools_by_names,
    format_tool_text,
)


def test_sentence_transformer_shortlister_structure():
    shortlister = SentenceTransformerToolShortlister()
    tools = load_tools()

    result = shortlister.shortlist(
        query="What is the current atmospheric weather condition in London?",
        tools=tools,
        expected_tool_name="get_weather",
        top_k=3
    )

    assert result.method_name == "sentence_transformers"
    assert len(result.tool_names) == 3
    assert result.expected_tool_present is True
    assert "get_weather" in result.tool_names
    assert result.retrieval_latency_ms >= 0.0
    assert len(result.scores) == 3
    assert all(isinstance(s, float) for s in result.scores.values())


def test_embedding_generation_and_shape():
    shortlister = SentenceTransformerToolShortlister()
    tools = load_tools()

    embeddings, names = shortlister.get_or_create_tool_embeddings(tools)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(tools)
    assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension
    assert len(names) == len(tools)


def test_embedding_cache_behavior():
    shortlister = SentenceTransformerToolShortlister()
    tools = load_tools()

    emb1, names1 = shortlister.get_or_create_tool_embeddings(tools)
    emb2, names2 = shortlister.get_or_create_tool_embeddings(tools)

    # Must be exact same cached object reference in memory
    assert emb1 is emb2
    assert names1 is names2


def test_retrieval_is_blind_to_ground_truth_labels():
    """
    Verify that the retrieval ranking algorithm produces identical rankings
    regardless of whether expected_tool_name is provided, None, or intentionally spoofed.
    """
    shortlister = SentenceTransformerToolShortlister()
    tools = load_tools()
    query = "Check whether my package TRK-88123 has arrived."

    res_blind = shortlister.shortlist(query=query, tools=tools, expected_tool_name=None, top_k=3)
    res_correct = shortlister.shortlist(query=query, tools=tools, expected_tool_name="track_shipment", top_k=3)
    res_spoofed = shortlister.shortlist(query=query, tools=tools, expected_tool_name="fake_nonexistent_tool", top_k=3)

    assert res_blind.tool_names == res_correct.tool_names
    assert res_blind.tool_names == res_spoofed.tool_names
    assert res_blind.scores == res_correct.scores


def test_deterministic_top_three_selection():
    """Verify that multiple consecutive calls return deterministic, identical order."""
    shortlister = SentenceTransformerToolShortlister()
    tools = load_tools()
    query = "How much will it cost to send a 20 kg package over 500 km?"

    res1 = shortlister.shortlist(query, tools, top_k=3)
    res2 = shortlister.shortlist(query, tools, top_k=3)

    assert res1.tool_names == res2.tool_names
    assert res1.scores == res2.scores


def test_sentence_transformer_top3_recall_on_benchmark():
    """Top-3 recall on the validated 30-task benchmark must exceed 90%."""
    shortlister = SentenceTransformerToolShortlister()
    tools = load_tools()
    tasks = load_tasks()

    hits = 0
    for task in tasks:
        res = shortlister.shortlist(
            query=task.paraphrased_query,
            tools=tools,
            expected_tool_name=task.expected_tool_name,
            top_k=3
        )
        if res.expected_tool_present:
            hits += 1

    recall = hits / len(tasks)
    assert recall >= 0.90, f"Expected top-3 recall >= 90%, got {recall*100:.1f}%"


def test_tfidf_shortlister_fallback():
    shortlister = TfidfToolShortlister()
    tools = load_tools()

    result = shortlister.shortlist(
        query="What is the current atmospheric weather in London?",
        tools=tools,
        expected_tool_name="get_weather",
        top_k=3
    )

    assert result.method_name == "tfidf"
    assert len(result.tool_names) == 3
    assert "get_weather" in result.tool_names


def test_filter_tools_by_names():
    tools = load_tools()
    allowed = ["get_weather", "calculate", "search_products"]
    filtered = filter_tools_by_names(tools, allowed)

    assert len(filtered) == 3
    assert [t.name for t in filtered] == allowed
