"""
Chapter 12 - Semantic cache.

Customer-support traffic repeats itself ("refund kab milega?", "when will I get my refund?").
A semantic cache stores answers by the MEANING of the question: if a new question's embedding is
very close to a cached one, return the cached answer and skip retrieval + LLM.

Two rules that people forget:
  * SCOPE the cache by permission and index version. An answer built from HR-only documents must
    never be served to a customer. When documents change, old answers must stop matching.
  * Keep the threshold HIGH (0.9+ for most models) and measure false hits on real traffic.
    "How do I cancel my order?" and "How do I cancel my membership?" are close but different.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class _Entry:
    vector: np.ndarray
    question: str
    answer: object
    created: float


class SemanticCache:
    def __init__(self, embedder, threshold: float = 0.92, ttl_seconds: int = 3600, max_items: int = 5000):
        self.embedder = embedder
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: dict[str, list[_Entry]] = {}
        self.hits = self.misses = 0

    @staticmethod
    def scope(user_groups, index_version: str) -> str:
        return f"{','.join(sorted(user_groups))}|{index_version}"

    def get(self, question: str, scope: str):
        now = time.time()
        entries = [e for e in self._store.get(scope, []) if now - e.created < self.ttl]
        self._store[scope] = entries
        if not entries:
            self.misses += 1
            return None
        q = self.embedder.embed_query(question)
        sims = np.array([e.vector @ q for e in entries])
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            self.hits += 1
            return entries[best].answer, float(sims[best]), entries[best].question
        self.misses += 1
        return None

    def put(self, question: str, scope: str, answer) -> None:
        entries = self._store.setdefault(scope, [])
        entries.append(_Entry(self.embedder.embed_query(question), question, answer, time.time()))
        if len(entries) > self.max_items:
            entries.pop(0)
