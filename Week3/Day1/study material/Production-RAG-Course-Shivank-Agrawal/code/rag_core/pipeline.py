"""
Everything wired together: the RAG pipeline used in Labs 09-13 and as a reference design.

INGEST (offline):  load -> clean -> redact PII -> chunk -> flag prompt injection -> embed -> index
QUERY  (online):   cache? -> filters (ACL, archived, injection) -> hybrid search -> rerank
                   -> threshold -> build context -> LLM (or extractive) -> citations -> trace
"""
from __future__ import annotations

from .cache import SemanticCache
from .chunking import chunk_documents
from .config import Settings, get_settings
from .documents import Document
from .embeddings import Embedder, fingerprint, get_embedder
from .generate import Answer, answer
from .hybrid import HybridRetriever, Hit
from .llm import LLM
from .loaders import load_corpus
from .observability import Tracer
from .rerank import get_reranker
from .security import acl_filter, detect_injection, redact_pii


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None, llm: LLM | None = None,
                 reranker="auto", chunk_strategy: str = "structure_aware", chunk_params: dict | None = None,
                 mode: str = "hybrid", k: int = 5, candidates: int = 30, redact: bool = True,
                 scan_injection: bool = True, cache: SemanticCache | None = None, tracer: Tracer | None = None):
        self.s = settings or get_settings()
        self.embedder = embedder or get_embedder()
        self.llm = llm if llm is not None else LLM(self.s)
        self.reranker = get_reranker() if reranker == "auto" else reranker  # None = no reranking
        self.chunk_strategy, self.chunk_params = chunk_strategy, chunk_params or {}
        self.mode, self.k, self.candidates = mode, k, candidates
        self.redact, self.scan_injection = redact, scan_injection
        self.cache = cache
        self.tracer = tracer or Tracer(self.s.log_dir / "traces.jsonl")
        self.retriever = HybridRetriever(self.embedder)
        self.index_version = "empty"
        self.ingest_report: dict = {}

    # ------------------------------------------------------------------ ingest
    def build(self, docs: list[Document] | None = None) -> "RAGPipeline":
        docs = docs if docs is not None else load_corpus()
        pii_total, flagged = 0, []
        if self.redact:
            for d in docs:
                d.text, findings = redact_pii(d.text)
                pii_total += len(findings)
        chunks = chunk_documents(docs, self.chunk_strategy, embedder=self.embedder, llm=self.llm, **self.chunk_params)
        for c in chunks:
            hits = detect_injection(c.text) if self.scan_injection else []
            c.metadata["injection_flag"] = bool(hits)
            if hits:
                flagged.append((c.chunk_id, hits))
        self.retriever.index(chunks)
        self.index_version = f"{fingerprint(self.embedder)}|{self.chunk_strategy}|{len(chunks)}"
        self.ingest_report = {"documents": len(docs), "chunks": len(chunks), "pii_redacted": pii_total,
                              "injection_flagged": flagged}
        return self

    # ------------------------------------------------------------------ query
    def filters(self, user_groups, include_archived: bool = False, extra: dict | None = None) -> dict:
        where = acl_filter(user_groups)
        where["injection_flag"] = {"$ne": True}  # quarantined chunks are never retrieved
        if not include_archived:
            where["status"] = {"$ne": "archived"}
        if extra:
            where.update(extra)
        return where

    def retrieve(self, question: str, user_groups=("public",), include_archived: bool = False,
                 extra_filters: dict | None = None, trace=None) -> list[Hit]:
        where = self.filters(user_groups, include_archived, extra_filters)
        if trace:
            with trace.span("retrieve", mode=self.mode) as sp:
                hits = self.retriever.retrieve(question, k=self.candidates, mode=self.mode, where=where,
                                               candidates=self.candidates)
                sp["n_candidates"] = len(hits)
        else:
            hits = self.retriever.retrieve(question, k=self.candidates, mode=self.mode, where=where,
                                           candidates=self.candidates)
        if self.reranker is not None and hits:
            if trace:
                with trace.span("rerank", model=self.reranker.name) as sp:
                    hits = self.reranker.rerank(question, hits, top_n=self.k)
                    sp["top_score"] = round(hits[0].rerank_score or 0, 3) if hits else None
            else:
                hits = self.reranker.rerank(question, hits, top_n=self.k)
        return hits[: self.k]

    def ask(self, question: str, user_groups=("public",), include_archived: bool = False,
            extra_filters: dict | None = None) -> Answer:
        trace = self.tracer.start("ask", question=question, groups=list(user_groups))
        scope = SemanticCache.scope(user_groups, self.index_version)
        if self.cache:
            with trace.span("cache_lookup") as sp:
                cached = self.cache.get(question, scope)
                sp["hit"] = cached is not None
            if cached:
                trace.set(cache_hit=True)
                trace.end()
                return cached[0]
        hits = self.retrieve(question, user_groups, include_archived, extra_filters, trace=trace)
        with trace.span("generate", provider=self.llm.provider) as sp:
            result = answer(question, hits, self.llm, min_similarity=self.s.min_similarity)
            sp.update(mode=result.mode, abstained=result.abstained, context_tokens=result.context_tokens)
            if result.usage:
                sp.update(input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens)
        trace.set(cache_hit=False, citations=[c["chunk_id"] for c in result.citations])
        trace.end()
        if self.cache and not result.abstained:
            self.cache.put(question, scope, result)
        return result
