"""
graph.py — the orchestrator: one graph, two engines.

    question ─► guard_input ─► route ─┬─► policy_qa ─────────────────────────────► answer
                                      ├─► vendor_research ─► facts_only ─────────► answer
                                      ├─► vendor_research ─► match ─► questionnaire ─► human_gate ─► report
                                      └─► refuse

The SAME node functions run either as plain Python (`Pipeline.run_plain`, always available) or as a
LangGraph StateGraph (`Pipeline.build_langgraph`, when langgraph is installed). Deep Search course Lab 12
does exactly this for the research loop; here it is for the whole product.

COURSE MAP
  Deep Search  Module 2 (workflows vs agents; the five patterns), Lab 12 (LangGraph version)
  Agentic RAG  "tasks.py" (how information flows from task to task — here it flows through `state`)

LEARNER TODO — Milestone 5
  * Add a checkpointer and `interrupt()` at human_gate so a run can pause for days and resume when the
    approver clicks (LangGraph docs: human-in-the-loop). The CLI version below asks on the terminal.
  * Stream node progress to the UI (app/ui.py) so a 60-second live run shows what it is doing.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any, Optional, TypedDict

from .config import SETTINGS
from .graders import grade_policy_answer
from .guardrails import check_intent, enforce_output_policy, redact_input
from .llm import LLM
from .matcher import Matcher
from .policy_rag import PolicyLibrary
from .questionnaire import as_email, build_questions
from .report import to_markdown
from .router import decide
from .schemas import Approval, AssessmentReport, AssessmentRequest
from .tracing import Budget, Tracer
from .vendor_research import VendorResearcher

APPROVERS = {"T1": ["Team lead"], "T2": ["Head of Information Security"], "T3": ["Head of Information Security", "DPO"]}


class State(TypedDict, total=False):
    """The shared workspace every node reads from and writes to (LangGraph merges updates per key)."""

    question: str
    user_groups: list[str]
    tier: str
    users: int
    vendor: Optional[str]
    domain: Optional[str]
    requested_by: str
    interactive: bool
    approver: Optional[str]
    decision: Optional[str]
    route: Optional[str]
    refusal: Optional[str]
    policy_answer: Any
    profile: Any
    request: Any
    results: Any
    overall: Optional[str]
    overall_reason: Optional[str]
    open_questions: list[dict]
    approval: Any
    report: Any
    answer: Optional[str]
    guardrail_notes: list[str]
    started: float


def new_state(question: str, *, user_groups: list[str] | None = None, tier: str = "T3", users: int = 1,
              vendor: str | None = None, domain: str | None = None, requested_by: str = "cli",
              interactive: bool = False, approver: str | None = None, decision: str | None = None) -> dict:
    return {"question": question, "user_groups": user_groups or ["public", "employees"], "tier": tier, "users": users,
            "vendor": vendor, "domain": domain, "requested_by": requested_by, "interactive": interactive,
            "approver": approver, "decision": decision,
            "route": None, "refusal": None, "policy_answer": None, "profile": None, "results": None, "overall": None,
            "overall_reason": None, "open_questions": [], "approval": None, "report": None, "answer": None,
            "guardrail_notes": [], "started": time.time()}


class Pipeline:
    def __init__(self, tracer: Tracer | None = None, budget: Budget | None = None, library: PolicyLibrary | None = None):
        self.tracer = tracer
        self.budget = budget or Budget()
        self.library = library or PolicyLibrary()
        self.matcher = Matcher(self.library)
        self.researcher = VendorResearcher(budget=self.budget, tracer=tracer)
        self.llm = LLM(budget=self.budget, tracer=tracer)
        self.known_vendors = list(self.researcher.web.vendors)

    def _t(self, stage: str, **data: Any) -> None:
        if self.tracer:
            self.tracer.event(stage, **data)

    # ------------------------------------------------------------------ nodes
    def guard_input(self, s: dict) -> dict:
        q, notes = redact_input(s["question"])
        refusal = check_intent(q)
        self._t("guard_input", redactions=notes, refused=bool(refusal))
        return {"question": q, "guardrail_notes": s["guardrail_notes"] + notes, "refusal": refusal,
                "route": "refuse" if refusal else None}

    def route(self, s: dict) -> dict:
        if s.get("route") == "refuse":
            return {}
        if s.get("vendor"):
            d = decide(s["question"], self.known_vendors + [s["vendor"]], self.llm)
            d.vendor = s["vendor"]
        else:
            d = decide(s["question"], self.known_vendors, self.llm)
        self._t("route", route=d.route, vendor=d.vendor, reason=d.reason)
        return {"route": d.route, "vendor": d.vendor or s.get("vendor")}

    def policy_qa(self, s: dict) -> dict:
        pa = self.library.answer(s["question"], s["user_groups"], llm=self.llm)
        pa = grade_policy_answer(pa, self.llm)
        self._t("policy_qa", abstained=pa.abstained, citations=[c.clause_id or c.heading for c in pa.citations[:3]],
                relevant=pa.grader_relevant, grounded=pa.grader_grounded)
        cites = "\n".join(f"  - {c.clause_id or c.heading} — {c.title} (v{c.version})" for c in pa.citations[:3])
        return {"policy_answer": pa, "answer": pa.answer + ("\n\nSources:\n" + cites if cites else "")}

    def vendor_research(self, s: dict) -> dict:
        profile = self.researcher.research(s["vendor"], s.get("domain"))
        return {"profile": profile}

    def facts_only(self, s: dict) -> dict:
        pf = s["profile"]
        lines = [f"What public sources say about {pf.vendor} (as of {pf.as_of}):"]
        for f, fact in pf.facts.items():
            lines.append(f"- {f}: {fact.display}  [{fact.evidence_level}, confidence {fact.confidence:.2f}, {fact.sources[0] if fact.sources else 'n/a'}]")
        if pf.quarantined:
            lines.append(f"- {len(pf.quarantined)} page(s) quarantined for prompt injection and ignored.")
        lines.append("Note: these are facts, not a policy verdict. Ask 'Can <team> use <tool>...' for an assessment.")
        return {"answer": "\n".join(lines)}

    def match(self, s: dict) -> dict:
        req = AssessmentRequest(question=s["question"], vendor=s["vendor"], official_domain=s["profile"].official_domain,
                                tier=s["tier"], users=s["users"], user_groups=s["user_groups"], requested_by=s["requested_by"])
        results = self.matcher.evaluate(s["profile"], req)
        overall, reason = self.matcher.overall(results)
        self._t("match", overall=overall, verdicts={r.requirement_id: r.verdict for r in results})
        return {"request": req, "results": results, "overall": overall, "overall_reason": reason}

    def questionnaire(self, s: dict) -> dict:
        qs = build_questions(s["results"], self.library)
        self._t("questionnaire", questions=[q["id"] for q in qs])
        return {"open_questions": qs}

    def human_gate(self, s: dict) -> dict:
        """A person decides. Offline/CI: status stays 'pending'. Interactive CLI: ask on the terminal."""
        roles = APPROVERS.get(s["tier"], [])
        approval = Approval(status="pending", approver_roles=roles)
        decision = s.get("decision")
        if s.get("interactive") and decision is None:
            print(f"\nOutcome so far: {s['overall']} — {s['overall_reason']}")
            print(f"Approvers for {s['tier']}: {', '.join(roles)}")
            decision = input("Record a decision now? [approve / reject / leave pending]: ").strip().lower() or "pending"
        if decision in ("approve", "approved") and s["overall"] != "BLOCKED":
            approval = Approval(status="approved", approver_roles=roles, decided_by=s.get("approver") or "cli-user", decided_on=date.today(),
                                comment="Recorded via CLI")
        elif decision in ("approve", "approved") and s["overall"] == "BLOCKED":
            approval.comment = "Approval refused by the gate: a blocking requirement failed. Resolve it or change the data tier."
        elif decision in ("reject", "rejected"):
            approval = Approval(status="rejected", approver_roles=roles, decided_by=s.get("approver") or "cli-user", decided_on=date.today())
        self._t("human_gate", status=approval.status, decided_by=approval.decided_by)
        return {"approval": approval}

    def report(self, s: dict) -> dict:
        rep = AssessmentReport(request=s["request"], profile=s["profile"], results=s["results"], overall=s["overall"],
                               overall_reason=s["overall_reason"], open_questions=s["open_questions"], approval=s["approval"],
                               guardrail_notes=list(s["guardrail_notes"]), run_id=self.tracer.run_id if self.tracer else "",
                               cost_usd=round(self.budget.spent_usd, 4), llm_calls=self.budget.llm_calls,
                               duration_s=round(time.time() - s["started"], 2))
        before = [r.verdict for r in rep.results]
        rep = enforce_output_policy(rep)
        if [r.verdict for r in rep.results] != before:   # a downgrade changes the outcome too
            rep.overall, rep.overall_reason = self.matcher.overall(rep.results)
        md = to_markdown(rep)
        if self.tracer:
            self.tracer.write_text("report.md", md)
            self.tracer.write_json("report.json", rep.model_dump(mode="json"))
            self.tracer.write_json("profile.json", rep.profile.model_dump(mode="json"))
            email = as_email(rep.request.vendor, rep.open_questions)
            if email:
                self.tracer.write_text("vendor_questions.txt", email)
        self._t("report", overall=rep.overall, run_dir=str(self.tracer.dir) if self.tracer else None)
        return {"report": rep, "answer": md}

    def refuse(self, s: dict) -> dict:
        return {"answer": s.get("refusal") or "I can only help with vendor assessments and policy questions."}

    # ------------------------------------------------------------------ engines
    def run_plain(self, s: dict) -> dict:
        """Plain Python. Read this first; it is the whole product in 15 lines."""
        s.update(self.guard_input(s))
        s.update(self.route(s))
        if s["route"] == "refuse":
            s.update(self.refuse(s))
        elif s["route"] == "policy":
            s.update(self.policy_qa(s))
        elif s["route"] == "vendor":
            s.update(self.vendor_research(s))
            s.update(self.facts_only(s))
        else:  # assessment
            s.update(self.vendor_research(s))
            s.update(self.match(s))
            s.update(self.questionnaire(s))
            s.update(self.human_gate(s))
            s.update(self.report(s))
        return s

    def build_langgraph(self):
        """The same nodes as a LangGraph state machine. Requires: pip install langgraph."""
        from langgraph.graph import END, START, StateGraph  # noqa: WPS433

        g = StateGraph(State)
        for name in ("guard_input", "route", "policy_qa", "vendor_research", "facts_only", "match", "questionnaire",
                     "human_gate", "report", "refuse"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "guard_input")
        g.add_edge("guard_input", "route")
        g.add_conditional_edges("route", lambda s: s["route"], {
            "refuse": "refuse", "policy": "policy_qa", "vendor": "vendor_research", "assessment": "vendor_research"})
        g.add_conditional_edges("vendor_research", lambda s: "facts_only" if s["route"] == "vendor" else "match",
                                {"facts_only": "facts_only", "match": "match"})
        g.add_edge("match", "questionnaire")
        g.add_edge("questionnaire", "human_gate")
        g.add_edge("human_gate", "report")
        for leaf in ("policy_qa", "facts_only", "report", "refuse"):
            g.add_edge(leaf, END)
        return g.compile()

    def run(self, s: dict, engine: str = "auto") -> dict:
        if engine in ("auto", "langgraph"):
            try:
                graph = self.build_langgraph()
                self._t("engine", name="langgraph")
                out = graph.invoke(s)
                s.update(out)
                return s
            except ImportError:
                if engine == "langgraph":
                    raise
        self._t("engine", name="plain")
        return self.run_plain(s)
