"""
orchestrator.py — the lead agent that runs the whole deep-search loop.

   question
      │
      ▼
   PLAN (create → review)
      │
      ▼
 ┌─► EXECUTE  (parallel workers: search → fetch → scan → extract)
 │    │
 │    ▼
 │  VALIDATE (identity, authority, freshness, copies, quarantine)
 │    │
 │    ▼
 │  MERGE    (dedupe → conflicts → provenance)
 │    │
 │    ▼
 └── EVALUATE (continue? stop?)  ──stop──►  SYNTHESISE final report → save + learn
"""

from __future__ import annotations

import json
import time

from .budget import Budget
from .config import SETTINGS
from .executor import Executor
from .llm import LLM
from .memory import Memory
from .merger import confidence_label, merge_all
from .planner import create_plan, evaluate_progress, review_plan
from .schemas import EntityAnchor, Evidence, FinalReport, Finding, MergedClaim, ResearchPlan
from .tracing import start_run, trace
from .validator import find_copies, score_evidence

FIELD_LABELS = {
    "ceo": "CEO", "founded_year": "Founded", "headquarters": "Headquarters", "latest_funding": "Latest funding",
    "regulatory_status": "Regulatory status", "pricing_domestic_cards": "Fee on domestic cards",
    "uptime_90d": "Uptime (90 days)", "major_outage": "Last major outage", "security_incident": "Security incident",
    "merchant_rating": "Merchant rating",
}

SYNTH_SYSTEM = """You write the final answer of a due-diligence research agent.
Use ONLY the findings JSON you are given — no outside knowledge. Write 4-7 sentences:
start with a direct answer to the question, then the key facts with their confidence, then
risks and open questions. Mention historical values only to explain changes. Cite sources
as [S1], [S2] using the source list given."""


def say(msg: str) -> None:
    if SETTINGS.verbose:
        print(msg, flush=True)


class DeepSearchAgent:
    def __init__(self, question: str, entity: EntityAnchor, budget: Budget | None = None,
                 memory: Memory | None = None, max_iterations: int | None = None, parallel: bool = True,
                 use_memory: bool = True, validation: bool = True):
        self.question = question
        self.entity = entity
        self.budget = budget or Budget()
        self.memory = memory or Memory()
        self.use_memory = use_memory
        self.max_iterations = max_iterations or SETTINGS.max_iterations
        self.parallel = parallel
        self.validation = validation  # False = ablation used in Lab 11 (no validator, no defences)
        self.llm = LLM(budget=self.budget)
        self.evidence: list[Evidence] = []
        self.plan: ResearchPlan | None = None
        self.iterations: list[dict] = []

    # ------------------------------------------------------------------
    def validate(self, executor: Executor) -> None:
        if not self.validation:  # ablation: trust everything equally, ignore identity and dates
            for ev in self.evidence:
                ev.accepted, ev.reliability = True, 0.6
            return
        copies = find_copies(list(executor.docs.values()))
        for ev in self.evidence:
            doc = executor.docs[ev.source.doc_id]
            ev.flags = [f for f in ev.flags if f not in ("identity_mismatch", "quarantined_source", "copy_of_other_page")]
            ev.accepted = True
            ev.origin_id = copies.get(ev.source.doc_id, "")
            if ev.origin_id:
                ev.flags.append("copy_of_other_page")
            score_evidence(ev, doc, self.entity, SETTINGS.as_of)
        trace("validate", evidence=len(self.evidence), accepted=sum(e.accepted for e in self.evidence),
              copies=copies)

    # ------------------------------------------------------------------
    def run(self) -> FinalReport:
        folder = start_run("deepsearch")
        started = time.time()
        say(f"\n▶ Question: {self.question}")
        if self.use_memory:
            for alike in self.memory.known_look_alikes(self.entity.name):
                if alike not in self.entity.exclude:
                    self.entity.exclude.append(alike)
        plan = create_plan(self.llm, self.question, self.entity, self.memory if self.use_memory else None)
        plan, review = review_plan(plan)
        plan.entity = self.entity
        self.plan = plan
        trace("plan", plan=plan.model_dump(), review=review.model_dump())
        say(f"▶ Plan: {len(plan.sub_questions)} sub-questions | reviewer fixes: {len(review.fixes_applied) - 1}")

        executor = Executor(self.entity, self.budget, max_parallel=SETTINGS.max_parallel,
                            results_per_query=SETTINGS.results_per_query, security=self.validation)
        if self.use_memory:
            executor.skip_domains = self.memory.blocked_domains()
            if executor.skip_domains:
                say(f"▶ Memory: skipping known-bad domains {sorted(executor.skip_domains)}")
        open_ids = [q.id for q in plan.sub_questions]
        merged, conflicts = {}, []
        for iteration in range(1, self.max_iterations + 1):
            todo = [q for q in plan.sub_questions if q.id in open_ids]
            say(f"▶ Iteration {iteration}: {len(todo)} worker(s) in {'parallel' if self.parallel else 'sequence'}")
            before = len(self.evidence)
            self.evidence.extend(executor.run(todo, iteration, parallel=self.parallel))
            new = len(self.evidence) - before
            self.validate(executor)
            merged, conflicts = merge_all(self.evidence, high_stakes_rule=self.validation)
            unresolved = {c.field for c in conflicts if c.resolution == "unresolved"}
            decision = evaluate_progress(plan, merged, unresolved, iteration, self.max_iterations, new, self.budget)
            self.iterations.append({"iteration": iteration, "new_evidence": new, "decision": decision.model_dump(),
                                    "budget": self.budget.summary()})
            trace("evaluate", iteration=iteration, decision=decision.model_dump())
            say(f"  new evidence: {new} | decision: {decision.action.upper()} — {decision.reasons[0] if decision.reasons else ''}")
            if decision.action == "stop":
                break
            open_ids = decision.open_sub_questions

        self.executor = executor  # kept so labs can inspect pages, quarantine and skips
        report = self.synthesise(merged, conflicts, executor, time.time() - started)
        self.learn(executor, conflicts)
        (folder / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (folder / "report.md").write_text(report_to_markdown(report), encoding="utf-8")
        (folder / "evidence.json").write_text(json.dumps([e.model_dump(mode="json") for e in self.evidence], indent=2),
                                              encoding="utf-8")
        say(f"▶ Saved report, evidence and trace to {folder}")
        self.run_folder = folder
        return report

    # ------------------------------------------------------------------
    def synthesise(self, merged: dict[str, list[MergedClaim]], conflicts, executor: Executor, seconds: float) -> FinalReport:
        sources: list[dict] = []
        url_to_sid: dict[str, str] = {}

        def sid(url: str) -> str:
            if url not in url_to_sid:
                url_to_sid[url] = f"S{len(url_to_sid) + 1}"
                doc = next((d for d in executor.docs.values() if d.meta.url == url), None)
                sources.append({"id": url_to_sid[url], "url": url, "title": doc.meta.title if doc else "",
                                "type": doc.meta.source_type if doc else "", "published": str(doc.meta.publication_date) if doc else ""})
            return url_to_sid[url]

        findings, history = [], []
        for field in FIELD_LABELS:
            for mc in merged.get(field, []):
                f = Finding(field=field, value=mc.value, confidence=confidence_label(mc.confidence), status=mc.status,
                            as_of=mc.as_of, sources=[sid(u) for u in mc.supporting_sources])
                if mc.status == "current":
                    findings.append(f)
                elif mc.status == "historical":
                    history.append(f)

        freshness = [f"Facts are as of {SETTINGS.as_of}. Evidence dates range "
                     f"{min((d.meta.publication_date for d in executor.docs.values() if d.meta.publication_date), default='n/a')} → "
                     f"{max((d.meta.last_updated or d.meta.publication_date for d in executor.docs.values() if d.meta.publication_date), default='n/a')}."]
        for f in findings:
            if f.field in ("uptime_90d", "pricing_domestic_cards") and f.as_of and (SETTINGS.as_of - f.as_of).days > 120:
                freshness.append(f"{FIELD_LABELS[f.field]} evidence is {(SETTINGS.as_of - f.as_of).days} days old — re-check before signing.")
        security = [f"Quarantined {q['url']} — {', '.join(q['findings'])}" for q in executor.quarantined]
        if executor.skipped:
            security.append("Skipped because memory flagged their domains (other company / injection): "
                            + ", ".join(sorted(set(executor.skipped))))
        rejected_identity = sorted({e.source.url for e in self.evidence if "identity_mismatch" in e.flags})
        if rejected_identity:
            security.append("Ignored pages about a different organisation: " + ", ".join(rejected_identity))
        answered = {f.field for f in findings}
        open_qs = [q.question for q in (self.plan.sub_questions if self.plan else [])
                   if not set(q.fields) & answered]
        open_qs += [f"Unresolved conflict on {FIELD_LABELS[c.field]}" for c in conflicts if c.resolution == "unresolved"]

        high = sum(f.confidence == "high" for f in findings)
        overall = "high" if findings and high / len(findings) >= 0.7 and not open_qs else "medium" if findings else "low"
        payload = {"question": self.question, "findings": [f.model_dump(mode="json") for f in findings],
                   "history": [h.model_dump(mode="json") for h in history],
                   "conflicts": [c.model_dump(mode="json") for c in conflicts], "open_questions": open_qs,
                   "sources": sources}
        answer = self.llm.chat(SYNTH_SYSTEM, json.dumps(payload, default=str)[:12000], stage="synthesise",
                               mock=lambda: template_answer(self.entity.name, findings, history, open_qs))
        return FinalReport(question=self.question, as_of=SETTINGS.as_of, answer=answer.strip(), confidence=overall,
                           key_findings=findings, historical_findings=history, conflicts=conflicts,
                           freshness_notes=freshness, security_notes=security, open_questions=open_qs, sources=sources,
                           run_stats={**self.budget.summary(), "iterations": len(self.iterations),
                                      "pages_read": len(executor.docs), "evidence_items": len(self.evidence),
                                      "wall_seconds": round(seconds, 2)})

    # ------------------------------------------------------------------
    def learn(self, executor: Executor, conflicts) -> None:
        if not self.use_memory:
            return
        for ev in self.evidence:
            reason = ",".join(ev.flags)
            self.memory.record_source(ev.source.domain, ev.accepted, reason)
        for q in executor.quarantined:
            self.memory.record_source(q["url"].split("/")[2], False, "prompt_injection")
        lessons: dict[str, str] = {}  # lesson text -> evidence (each lesson counted once per run)
        for ev in self.evidence:
            if "identity_mismatch" in ev.flags:
                self.memory.add_look_alike(self.entity.name, ev.source.domain)
                lessons["Same-name companies exist: require city/country/industry anchors before accepting a page."] = ev.source.url
        if executor.quarantined:
            lessons["SEO 'review' pages can carry prompt injections: never use them for facts."] = executor.quarantined[0]["url"]
        for c in conflicts:
            if c.resolution == "time_separated":
                lessons[f"'{c.field}' changes over time: search for the newest official source first."] = c.rationale[:120]
        if any("copy_of_other_page" in e.flags for e in self.evidence):
            lessons["Press releases get syndicated: count copies as one source."] = "find_copies()"
        for text, evidence in lessons.items():
            self.memory.add_lesson(text, evidence=evidence)
        self.memory.save()


def template_answer(name: str, findings: list[Finding], history: list[Finding], open_qs: list[str]) -> str:
    get = {f.field: f for f in findings}

    def v(field: str) -> str:
        f = get.get(field)
        return f"{f.value} ({f.confidence} confidence, {' '.join(f.sources)})" if f else "not established"

    parts = [
        f"As of the research date, {name} looks like a usable but not risk-free payments partner.",
        f"Regulatory status: {v('regulatory_status')}. CEO: {v('ceo')}. Latest funding: {v('latest_funding')}.",
        f"Reliability: uptime {v('uptime_90d')}; last major outage {v('major_outage')}.",
        f"Security: {v('security_incident')}. Pricing on domestic cards: {v('pricing_domestic_cards')}. "
        f"Merchant rating: {v('merchant_rating')}.",
    ]
    if history:
        parts.append("Changed over time: " + "; ".join(f"{FIELD_LABELS[h.field]} was {h.value}" for h in history) + ".")
    if open_qs:
        parts.append("Open questions: " + "; ".join(open_qs[:3]) + ".")
    return " ".join(parts)


def report_to_markdown(r: FinalReport) -> str:
    lines = ["# Deep search report\n", f"**Question:** {r.question}  ", f"**As of:** {r.as_of}  ",
             f"**Overall confidence:** {r.confidence}\n", "## Answer\n", r.answer, "\n## Key findings\n",
             "| Field | Value | Confidence | As of | Sources |", "|---|---|---|---|---|"]
    for f in r.key_findings:
        lines.append(f"| {FIELD_LABELS.get(f.field, f.field)} | {f.value} | {f.confidence} | {f.as_of or ''} | {', '.join(f.sources)} |")
    if r.historical_findings:
        lines += ["\n## History (older values)\n"] + [f"- {FIELD_LABELS.get(h.field, h.field)}: {h.value} (as of {h.as_of})"
                                                     for h in r.historical_findings]
    if r.conflicts:
        lines += ["\n## Conflicts and how they were resolved\n"] + [f"- **{FIELD_LABELS.get(c.field, c.field)}** → "
                                                                  f"`{c.resolution}`: {c.rationale}" for c in r.conflicts]
    for title, items in (("Freshness notes", r.freshness_notes), ("Security notes", r.security_notes),
                         ("Open questions", r.open_questions)):
        if items:
            lines += [f"\n## {title}\n"] + [f"- {x}" for x in items]
    lines += ["\n## Sources\n"] + [f"- [{s['id']}] {s['title']} — {s['url']} ({s['type']}, {s['published']})" for s in r.sources]
    lines += ["\n## Run stats\n", "```json", json.dumps(r.run_stats, indent=2), "```"]
    return "\n".join(lines)


def default_entity() -> EntityAnchor:
    """The running example of the course (all fictional)."""
    return EntityAnchor(name="NimbusPay", country="India", city="Bengaluru",
                        industry_keywords=["payment gateway", "payment aggregator", "payments", "merchants"],
                        official_domain="nimbuspay.example")


DEFAULT_QUESTION = ("Is NimbusPay (Bengaluru, India) a reliable payment-gateway partner for our D2C startup today? "
                    "Check leadership, licence status, funding, reliability, security incidents, pricing and merchant sentiment.")
