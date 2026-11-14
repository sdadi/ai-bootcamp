"""
baselines.py — the two systems deep search is meant to beat.

1. naive_search_answer  — search once, read the top snippets, answer.
2. basic_rag_answer     — retrieve top chunks for each topic, answer from them.

Both treat every chunk as equally trustworthy, ignore dates, never check
identity and never look for conflicts. Lab 1 and Lab 11 compare them with the
full deep-search agent on the same question.
"""

from __future__ import annotations

import re

from .extractor import gullible_extract, rule_extract
from .llm import LLM
from .schemas import Document, SourceMeta
from .tools import _BM25, corpus_docs, fetch_page, search_web

FIELDS_TO_TOPICS = {
    "ceo": "CEO", "founded_year": "founded", "headquarters": "headquartered",
    "latest_funding": "raised funding Series", "regulatory_status": "licence status",
    "pricing_domestic_cards": "per transaction domestic cards", "uptime_90d": "uptime",
    "major_outage": "outage", "security_incident": "security incident storage bucket",
    "merchant_rating": "rating reviews",
}


def _first_values(claim_lists) -> dict[str, str]:
    answer: dict[str, str] = {}
    for claims in claim_lists:
        for c in claims:
            answer.setdefault(c.field, c.value)  # first come, first served — no validation at all
    return answer


def naive_search_answer(question: str, top_k: int = 3, llm: LLM | None = None) -> dict:
    """One search with the user's question, read the top results, take whatever they say."""
    results = search_web(question, max_results=top_k)
    claim_lists, used = [], []
    for r in results:
        doc = fetch_page(r.url)
        if doc:
            claim_lists.append(gullible_extract(doc).claims)  # no injection defence, no identity check
            used.append(r.url)
    return {"answers": _first_values(claim_lists), "sources": used}


def _chunks() -> list[dict]:
    """Split every corpus page into sentence chunks (the classic RAG pre-processing step)."""
    chunks = []
    for d in corpus_docs():
        raw = d["text"].replace("<!--", ". ").replace("-->", ". ")  # naive HTML handling keeps hidden text
        for i, sent in enumerate(re.split(r"(?<=[.!?])\s+", raw)):
            if len(sent) > 20:
                chunks.append({"id": f"{d['id']}-{i}", "title": d["title"], "text": sent, "url": d["url"], "doc": d})
    return chunks


def basic_rag_answer(question: str, top_k: int = 2) -> dict:
    """For each topic, retrieve the top-k chunks from a fixed index and extract from them."""
    chunks = _chunks()
    engine = _BM25([{"title": "", "text": c["text"], **c} for c in chunks])
    claim_lists, used = [], set()
    for field, topic in FIELDS_TO_TOPICS.items():
        for chunk, _score in engine.search(f"NimbusPay {topic}", top_k):
            d = chunk["doc"]
            meta = SourceMeta(doc_id=d["id"], url=d["url"], domain="", title=d["title"], source_type=d["source_type"],
                              retrieval_date="2026-11-01")
            claims = [c for c in gullible_extract(Document(meta=meta, text=chunk["text"])).claims if c.field == field]
            if claims:
                claim_lists.append(claims)
                used.add(d["url"])
    return {"answers": _first_values(claim_lists), "sources": sorted(used)}


__all__ = ["naive_search_answer", "basic_rag_answer", "rule_extract"]
