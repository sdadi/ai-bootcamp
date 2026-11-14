"""
executor.py — turns the plan into structured evidence.

Orchestrator-worker pattern: each open sub-question gets its own WORKER
("subagent") that writes queries, searches, fetches pages, scans them for
prompt injection and extracts claims. Workers run IN PARALLEL (asyncio), so a
plan with 8 sub-questions takes roughly as long as the slowest worker instead
of the sum of all of them.

Every worker returns Evidence objects — never free text — so the validator and
merger can work on clean, typed data.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

from pydantic import BaseModel, Field

from .budget import Budget
from .extractor import gullible_extract, llm_extract
from .llm import LLM
from .schemas import Document, EntityAnchor, Evidence, SubQuestion
from .security import is_suspicious, scan_for_injection
from .tools import fetch_page, search_web
from .tracing import trace

TOPIC_WORDS = {
    "ceo": "CEO chief executive",
    "founded_year": "founded about",
    "headquarters": "headquartered about",
    "latest_funding": "funding raised Series",
    "regulatory_status": "payment aggregator licence authorisation registry",
    "pricing_domestic_cards": "pricing per transaction domestic cards",
    "uptime_90d": "uptime status",
    "major_outage": "outage down incident",
    "security_incident": "security incident data exposure",
    "merchant_rating": "reviews rating merchants",
}
SECOND_PASS_WORDS = {
    "ceo": "appoints new CEO news",
    "regulatory_status": "regulator registry final authorisation",
    "latest_funding": "investors round analysis",
    "uptime_90d": "system status 90 days",
    "major_outage": "status page incident",
    "security_incident": "security notice",
    "pricing_domestic_cards": "official pricing page",
    "merchant_rating": "merchant reviews complaints",
    "founded_year": "company profile",
    "headquarters": "company profile",
}


class QueryList(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=4)


QUERY_SYSTEM = """You write web search queries for a research worker.
Write 2 specific queries for the sub-question. Always include identity anchors (company name plus
city or country or industry) so results are about the right company. On later iterations, try a
DIFFERENT angle from the queries already used (e.g. regulator registry, official site, news)."""


def make_queries(llm: LLM, sq: SubQuestion, entity: EntityAnchor, iteration: int, used: list[str]) -> list[str]:
    def mock() -> QueryList:
        words = SECOND_PASS_WORDS if iteration > 1 else TOPIC_WORDS
        qs = [f"{entity.name} {entity.city or entity.country} {words[f]}".strip() for f in sq.fields]
        return QueryList(queries=[q for q in qs if q not in used] or [f"{entity.name} {sq.question[:60]}"])
    user = f"Sub-question: {sq.question}\nEntity: {entity.model_dump_json()}\nIteration: {iteration}\nAlready used: {used}"
    return llm.structured(QUERY_SYSTEM, user, QueryList, stage="query_gen", mock=mock).queries


def _simulated_latency() -> None:
    delay = float(os.getenv("DS_SIMULATED_LATENCY", "0"))
    if delay:
        time.sleep(delay)


class Executor:
    def __init__(self, entity: EntityAnchor, budget: Budget, max_parallel: int = 4, results_per_query: int = 5,
                 security: bool = True):
        self.entity = entity
        self.budget = budget
        self.max_parallel = max_parallel
        self.results_per_query = results_per_query
        self.security = security  # False = ablation: no injection scan, gullible extraction (Lab 11)
        self.planner_llm = LLM(budget=budget)
        self.extract_llm = LLM.small(budget=budget)  # model routing: cheaper model for extraction
        self.seen_urls: set[str] = set()
        self.docs: dict[str, Document] = {}
        self.quarantined: list[dict] = []
        self.queries_used: dict[str, list[str]] = {}
        self.skip_domains: set[str] = set()  # filled from long-term memory
        self.skipped: list[str] = []
        self._lock = threading.Lock()
        self._counter = 0

    # ---- one worker ---------------------------------------------------------
    def run_worker(self, sq: SubQuestion, iteration: int) -> list[Evidence]:
        used = self.queries_used.setdefault(sq.id, [])
        queries = make_queries(self.planner_llm, sq, self.entity, iteration, used)
        used.extend(queries)
        evidence: list[Evidence] = []
        for q in queries:
            if self.budget.exhausted():
                break
            _simulated_latency()
            for result in search_web(q, max_results=self.results_per_query, budget=self.budget):
                with self._lock:
                    if result.url in self.seen_urls:
                        continue
                    self.seen_urls.add(result.url)
                    if result.domain in self.skip_domains:  # memory says: not worth a fetch
                        self.skipped.append(result.url)
                        trace("skip_from_memory", url=result.url)
                        continue
                _simulated_latency()
                doc = fetch_page(result.url, budget=self.budget)
                if doc is None:
                    continue
                od = self.entity.official_domain
                if od and (doc.meta.domain == od or doc.meta.domain.endswith("." + od)):
                    doc.meta.source_type = "official"  # live mode: we know the company's own site
                findings = scan_for_injection(doc.text) if self.security else []
                if is_suspicious(findings):
                    doc.flags.append("prompt_injection_suspected")
                    with self._lock:
                        self.quarantined.append({"url": doc.meta.url, "findings": findings})
                    trace("quarantine", url=doc.meta.url, findings=findings)
                with self._lock:
                    self.docs[doc.meta.doc_id] = doc
                if "prompt_injection_suspected" in doc.flags:
                    continue  # never send a suspected attack page to the extractor
                if self.security:
                    claims = llm_extract(self.extract_llm, doc, self.entity, sq.question).claims
                else:
                    claims = gullible_extract(doc).claims
                for c in claims:
                    with self._lock:
                        self._counter += 1
                        ev_id = f"e{self._counter:03d}"
                    evidence.append(Evidence(id=ev_id, sub_question_id=sq.id, claim=c, source=doc.meta))
        trace("worker_done", sub_question=sq.id, iteration=iteration, queries=queries, evidence=len(evidence))
        return evidence

    # ---- fan-out ---------------------------------------------------------
    async def _run_all_async(self, sub_questions: list[SubQuestion], iteration: int) -> list[Evidence]:
        sem = asyncio.Semaphore(self.max_parallel)

        async def guarded(sq: SubQuestion) -> list[Evidence]:
            async with sem:
                return await asyncio.to_thread(self.run_worker, sq, iteration)

        batches = await asyncio.gather(*(guarded(sq) for sq in sub_questions))
        return [ev for batch in batches for ev in batch]

    def run(self, sub_questions: list[SubQuestion], iteration: int, parallel: bool = True) -> list[Evidence]:
        if parallel and self.max_parallel > 1:
            return asyncio.run(self._run_all_async(sub_questions, iteration))
        out: list[Evidence] = []
        for sq in sub_questions:
            out.extend(self.run_worker(sq, iteration))
        return out
