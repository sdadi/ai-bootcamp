"""
Chapter 8 - BM25 keyword search, written from scratch (about 40 lines).

BM25 scores a chunk higher when:
  * it contains the query words (term frequency, with diminishing returns: k1),
  * those words are RARE in the whole collection (inverse document frequency),
  * the chunk is not unusually long (length normalisation: b).

Why we need it next to embeddings: identifiers like "E102", "NM-PHN-4821" or
"ERR_UPI_PENDING" carry almost no *meaning* for an embedding model, but BM25 matches
them exactly.
"""
from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i if in is it its my of on or "
    "our so that the their there this to was we what when where which who why will with you your "
    "me am did any".split()
)

# Keep codes together: "nm-phn-4821", "err_upi_pending", "e102", "₹5000"
_TOKEN = re.compile(r"[a-z0-9₹]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())  # "5,000" -> "5000"
    return [t for t in _TOKEN.findall(text) if t not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        # the "+1" inside the log keeps idf positive even for very common words
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def score(self, query: str, index: int) -> float:
        doc, length = self.docs[index], self.lengths[index]
        s = 0.0
        for term in tokenize(query):
            tf = doc.get(term, 0)
            if tf:
                s += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
        return s

    def search(self, query: str, k: int = 5, allowed: list[int] | None = None) -> list[tuple[int, float]]:
        candidates = range(len(self.docs)) if allowed is None else allowed
        scored = [(i, self.score(query, i)) for i in candidates]
        scored = [x for x in scored if x[1] > 0]
        return sorted(scored, key=lambda x: -x[1])[:k]
