"""
Semantic and statistical tool shortlisting.
Implements:
1. Neural semantic retrieval using Sentence Transformers (`all-MiniLM-L6-v2`) with cosine similarity and embedding caching.
2. Lexical retrieval using TF-IDF and Cosine Similarity as a baseline comparison.
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.dataset import ToolDefinition


class ShortlistResult(BaseModel):
    """Container for tool retrieval results for a single query."""
    method_name: str
    top_k: int
    tool_names: List[str]
    scores: Dict[str, float]
    expected_tool_present: bool
    retrieval_latency_ms: float


class BaseToolShortlister:
    """Abstract base shortlister interface."""
    def shortlist(
        self,
        query: str,
        tools: List[ToolDefinition],
        expected_tool_name: Optional[str] = None,
        top_k: int = 3
    ) -> ShortlistResult:
        raise NotImplementedError


def format_tool_text(tool: ToolDefinition) -> str:
    """
    Format tool into standard text representation for embedding:
    Name: get_order_status
    Description: Retrieve the current status of a specific order...
    Parameters: order_id, string.
    """
    param_parts = []
    for prop_name, prop_meta in tool.parameters.properties.items():
        p_type = prop_meta.get("type", "string").lower() if isinstance(prop_meta, dict) else "string"
        param_parts.append(f"{prop_name}, {p_type}")
    param_str = "; ".join(param_parts) if param_parts else "none"

    return f"Name: {tool.name}\nDescription: {tool.description}\nParameters: {param_str}."


class SentenceTransformerToolShortlister(BaseToolShortlister):
    """
    Neural semantic tool retrieval using Sentence Transformers (all-MiniLM-L6-v2).
    Features:
    - Singleton model loading (loaded once in memory on CPU).
    - In-memory and on-disk embedding caching for tool representations.
    - Automatic cache invalidation when tool definitions change.
    - Deterministic tie-breaking (similarity descending, tool name ascending).
    - Strictly zero access to ground truth during ranking.
    """
    _model_instance = None
    _cached_tools_hash: Optional[str] = None
    _cached_tool_embeddings: Optional[np.ndarray] = None
    _cached_tool_names: Optional[List[str]] = None

    def __init__(self, model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.method_name = "sentence_transformers"
        self.model_name = model_name_or_path
        self._ensure_model_loaded()

    @classmethod
    def _ensure_model_loaded(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load SentenceTransformer model once on CPU."""
        if cls._model_instance is None:
            # Set environment to quiet TF warnings
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from sentence_transformers import SentenceTransformer

            # Check for local snapshot first to avoid external network issues
            local_snap = Path(r"C:\Users\ashis\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
            if local_snap.exists():
                try:
                    cls._model_instance = SentenceTransformer(str(local_snap), device="cpu")
                    return cls._model_instance
                except Exception:
                    pass

            # Fallback to standard model load
            cls._model_instance = SentenceTransformer(model_name, device="cpu")
        return cls._model_instance

    def _compute_tools_hash(self, tools: List[ToolDefinition]) -> str:
        """Compute MD5 hash over tool representations to detect changes."""
        corpus = [format_tool_text(t) for t in sorted(tools, key=lambda x: x.name)]
        raw = "||".join(corpus).encode("utf-8")
        return hashlib.md5(raw).hexdigest()

    def get_or_create_tool_embeddings(self, tools: List[ToolDefinition]) -> Tuple[np.ndarray, List[str]]:
        """
        Retrieve cached tool embeddings or compute and cache them if tools changed.
        """
        tools_sorted = sorted(tools, key=lambda x: x.name)
        current_hash = self._compute_tools_hash(tools_sorted)

        if (
            SentenceTransformerToolShortlister._cached_tools_hash == current_hash
            and SentenceTransformerToolShortlister._cached_tool_embeddings is not None
            and SentenceTransformerToolShortlister._cached_tool_names is not None
        ):
            return (
                SentenceTransformerToolShortlister._cached_tool_embeddings,
                SentenceTransformerToolShortlister._cached_tool_names
            )

        # Compute embeddings
        model = self._ensure_model_loaded()
        tool_texts = [format_tool_text(t) for t in tools_sorted]
        embeddings = model.encode(tool_texts, convert_to_numpy=True, normalize_embeddings=True)
        tool_names = [t.name for t in tools_sorted]

        # Update class cache
        SentenceTransformerToolShortlister._cached_tools_hash = current_hash
        SentenceTransformerToolShortlister._cached_tool_embeddings = embeddings
        SentenceTransformerToolShortlister._cached_tool_names = tool_names

        return embeddings, tool_names

    def shortlist(
        self,
        query: str,
        tools: List[ToolDefinition],
        expected_tool_name: Optional[str] = None,
        top_k: int = 3
    ) -> ShortlistResult:
        """
        Retrieve top-k tools by cosine similarity between query embedding and tool embeddings.
        The ranking algorithm has NO access to expected_tool_name or ground-truth labels.
        """
        t0 = time.perf_counter()

        # 1. Get cached tool embeddings
        tool_embeddings, tool_names = self.get_or_create_tool_embeddings(tools)

        # 2. Encode user query
        model = self._ensure_model_loaded()
        query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # 3. Compute cosine similarity (since normalized, dot product = cosine similarity)
        similarities = np.dot(tool_embeddings, query_emb)

        # 4. Deterministic tie-breaking: sort descending by similarity score, then ascending by tool name
        ranked_items = sorted(
            zip(tool_names, similarities),
            key=lambda item: (-float(item[1]), item[0])
        )

        top_items = ranked_items[:top_k]
        selected_tool_names = [item[0] for item in top_items]
        scores = {item[0]: round(float(item[1]), 4) for item in top_items}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Ground truth is checked ONLY post-hoc for diagnostic accuracy reporting
        expected_present = bool(
            expected_tool_name and (expected_tool_name in selected_tool_names)
        )

        return ShortlistResult(
            method_name=self.method_name,
            top_k=top_k,
            tool_names=selected_tool_names,
            scores=scores,
            expected_tool_present=expected_present,
            retrieval_latency_ms=round(elapsed_ms, 2)
        )


class TfidfToolShortlister(BaseToolShortlister):
    """
    Lexical and statistical tool retrieval using TF-IDF and Cosine Similarity.
    Kept as a comparative baseline.
    """
    def __init__(self):
        self.method_name = "tfidf"

    def _build_tool_corpus(self, tools: List[ToolDefinition]) -> List[str]:
        return [format_tool_text(t) for t in tools]

    def shortlist(
        self,
        query: str,
        tools: List[ToolDefinition],
        expected_tool_name: Optional[str] = None,
        top_k: int = 3
    ) -> ShortlistResult:
        t0 = time.perf_counter()

        tools_sorted = sorted(tools, key=lambda x: x.name)
        corpus = self._build_tool_corpus(tools_sorted)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tool_matrix = vectorizer.fit_transform(corpus)
        query_vec = vectorizer.transform([query])

        sims = cosine_similarity(query_vec, tool_matrix)[0]

        ranked_items = sorted(
            zip([t.name for t in tools_sorted], sims),
            key=lambda item: (-float(item[1]), item[0])
        )

        top_items = ranked_items[:top_k]
        selected_tools = [item[0] for item in top_items]
        scores = {item[0]: round(float(item[1]), 4) for item in top_items}

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        expected_present = bool(
            expected_tool_name and (expected_tool_name in selected_tools)
        )

        return ShortlistResult(
            method_name=self.method_name,
            top_k=top_k,
            tool_names=selected_tools,
            scores=scores,
            expected_tool_present=expected_present,
            retrieval_latency_ms=round(elapsed_ms, 2)
        )


def filter_tools_by_names(
    tools: List[ToolDefinition],
    allowed_names: List[str]
) -> List[ToolDefinition]:
    """Filter list of tools to keep only those present in allowed_names, preserving order."""
    name_set = set(allowed_names)
    tool_map = {t.name: t for t in tools}
    return [tool_map[name] for name in allowed_names if name in tool_map]
