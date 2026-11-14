"""
Chapter 8 - Retrieval: vector search, keyword search, and how to combine them.

Modes of HybridRetriever.retrieve():
    "vector"          embeddings only
    "bm25"            keywords only
    "weighted_naive"  alpha * cosine + (1 - alpha) * raw BM25   <- the WRONG way (scales differ)
    "weighted_minmax" both scores min-max scaled to 0..1 first, then weighted
    "hybrid"          Reciprocal Rank Fusion (RRF) of the two rankings  <- recommended default

Why RRF? Cosine similarity lives in about -1..1 (for many models, 0.3..0.9), while BM25 scores
are unbounded (2, 9, 17...). Adding them directly lets BM25 dominate. RRF ignores the raw
scores and uses only the RANK of each chunk in each list:

    RRF(chunk) = sum over lists of 1 / (k + rank)      with k = 60 (Cormack et al., 2009)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bm25 import BM25
from .documents import Chunk
from .embeddings import Embedder, fingerprint
from .vectorstore import InMemoryVectorStore


@dataclass
class Hit:
    chunk: Chunk
    score: float                      # the score this list is sorted by
    vector_score: float | None = None  # cosine similarity (useful for "do I know this?" thresholds)
    bm25_score: float | None = None
    ranks: dict[str, int] = field(default_factory=dict)
    rerank_score: float | None = None


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion. Each ranking is a list of ids, best first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {key: 1.0 for key in scores}
    return {key: (v - lo) / (hi - lo) for key, v in scores.items()}


class HybridRetriever:
    def __init__(self, embedder: Embedder, chunks: list[Chunk] | None = None):
        self.embedder = embedder
        self.store = InMemoryVectorStore(embedder.dim, fingerprint(embedder))
        self.bm25: BM25 | None = None
        if chunks:
            self.index(chunks)

    def index(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self.bm25 = BM25([c.text for c in self.store.chunks])  # rebuild over everything

    @property
    def chunks(self) -> list[Chunk]:
        return self.store.chunks

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid", where: dict | None = None,
                 candidates: int = 30, alpha: float = 0.5, rrf_k: int = 60,
                 query_vector: np.ndarray | None = None) -> list[Hit]:
        allowed = self.store.allowed_indices(where)
        if len(allowed) == 0:
            return []
        qv = query_vector if query_vector is not None else self.embedder.embed_query(query)
        vec_all = self.store.vectors[allowed] @ qv
        vec_scores = {int(i): float(s) for i, s in zip(allowed, vec_all)}
        vec_rank = sorted(vec_scores, key=lambda i: -vec_scores[i])[:candidates]
        bm_list = self.bm25.search(query, k=candidates, allowed=allowed.tolist())
        bm_scores = {i: s for i, s in bm_list}
        bm_rank = [i for i, _ in bm_list]

        def hit(i: int, score: float) -> Hit:
            ranks = {}
            if i in vec_rank:
                ranks["vector"] = vec_rank.index(i) + 1
            if i in bm_rank:
                ranks["bm25"] = bm_rank.index(i) + 1
            return Hit(self.store.chunks[i], score, vec_scores.get(i),
                       bm_scores.get(i, self.bm25.score(query, i)), ranks)

        if mode == "vector":
            order = [(i, vec_scores[i]) for i in vec_rank]
        elif mode == "bm25":
            order = [(i, s) for i, s in bm_list]
        elif mode in {"weighted_naive", "weighted_minmax"}:
            union = set(vec_rank) | set(bm_rank)
            v = {i: vec_scores[i] for i in union}
            b = {i: bm_scores.get(i, self.bm25.score(query, i)) for i in union}
            if mode == "weighted_minmax":
                v, b = minmax(v), minmax(b)
            combined = {i: alpha * v[i] + (1 - alpha) * b[i] for i in union}
            order = sorted(combined.items(), key=lambda x: -x[1])
        elif mode == "hybrid":
            fused = rrf([[str(i) for i in vec_rank], [str(i) for i in bm_rank]], k=rrf_k)
            order = sorted(((int(i), s) for i, s in fused.items()), key=lambda x: -x[1])
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return [hit(i, s) for i, s in order[:k]]

    def retrieve_many(self, queries: list[str], k: int = 5, mode: str = "hybrid", where: dict | None = None,
                      candidates: int = 30) -> list[Hit]:
        """Run several queries (from multi-query or HyDE) and fuse all their rankings with RRF."""
        runs = [self.retrieve(q, k=candidates, mode=mode, where=where, candidates=candidates) for q in queries]
        fused = rrf([[h.chunk.chunk_id for h in run] for run in runs])
        best: dict[str, Hit] = {}
        for run in runs:
            for h in run:
                if h.chunk.chunk_id not in best or (h.vector_score or 0) > (best[h.chunk.chunk_id].vector_score or 0):
                    best[h.chunk.chunk_id] = h
        ordered = sorted(fused.items(), key=lambda x: -x[1])[:k]
        out = []
        for cid, s in ordered:
            h = best[cid]
            out.append(Hit(h.chunk, s, h.vector_score, h.bm25_score, h.ranks))
        return out
