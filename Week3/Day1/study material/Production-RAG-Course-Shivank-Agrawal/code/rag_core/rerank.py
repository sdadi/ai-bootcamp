"""
Chapter 8 - Reranking: the second, more careful look at the candidates.

Stage 1 (fast, recall):     hybrid search returns ~25-50 candidates.
Stage 2 (slow, precision):  a CROSS-ENCODER reads the query and each candidate TOGETHER
                            and scores how well the candidate answers the query.

A bi-encoder (embedding model) encodes query and chunk separately, so document vectors can
be pre-computed. A cross-encoder must run once per (query, chunk) pair at query time, so we
only use it on a short list.
"""
from __future__ import annotations

import time
import warnings

from .hybrid import Hit


class CrossEncoderReranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self.model = TextCrossEncoder(model_name=model_name)
        self.last_latency_ms = 0.0

    def rerank(self, query: str, hits: list[Hit], top_n: int = 5) -> list[Hit]:
        if not hits:
            return []
        t0 = time.perf_counter()
        scores = list(self.model.rerank(query, [h.chunk.text for h in hits]))
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: -(h.rerank_score or 0))[:top_n]


class KeywordOverlapReranker:
    """A model-free stand-in used ONLY when the cross-encoder cannot be downloaded.
    It rewards chunks that contain more of the query's words. Much weaker than a real reranker."""

    name = "keyword-overlap (fallback)"
    last_latency_ms = 0.0

    def rerank(self, query: str, hits: list[Hit], top_n: int = 5) -> list[Hit]:
        from .bm25 import tokenize

        q = set(tokenize(query))
        for h in hits:
            words = set(tokenize(h.chunk.text))
            h.rerank_score = len(q & words) / max(1, len(q)) + 0.1 * (h.vector_score or 0)
        return sorted(hits, key=lambda h: -(h.rerank_score or 0))[:top_n]


def get_reranker(model_name: str | None = None):
    from .config import get_settings

    name = model_name or get_settings().reranker_model
    try:
        return CrossEncoderReranker(name)
    except Exception as e:  # no internet / model not found
        warnings.warn(f"Could not load reranker {name!r} ({e.__class__.__name__}). "
                      "Using the keyword-overlap fallback instead.", stacklevel=2)
        return KeywordOverlapReranker()
