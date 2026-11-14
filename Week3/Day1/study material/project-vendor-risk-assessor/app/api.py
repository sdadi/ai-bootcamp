"""
app/api.py — the assessor as a web service (FastAPI).

COURSE MAP  Production RAG course Module 4 (entry points and validation), Module 13 (the production app:
api.py / worker.py pattern). This is the thin version: no queue, no database. Milestone 5 adds them.

Run:   uvicorn app.api:app --reload --port 8000     then open http://localhost:8000/docs

Endpoints
  POST /ask        {"question": "...", "user_groups": [...]}                 → policy answer with citations
  POST /assess     {"question": "...", "vendor": "...", "tier": "T3", "users": 35, "user_groups": [...]}
                                                                           → full report JSON (approval pending)
  POST /approve/{run_id}  {"decision": "approve"|"reject", "approver": "...", "comment": "..."}
                                                                           → records the human decision on the saved report
  GET  /runs/{run_id}/report.md                                            → the Markdown review file
  GET  /health

LEARNER TODO — Milestone 5
  * Put /assess behind a queue (Redis + RQ as in the Production RAG app) — live research takes a minute.
  * Store reports in Postgres instead of runs/ so the UI can list and compare past assessments (SEC-12.1).
  * Authenticate the caller and derive user_groups from their identity — never trust groups from the client.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from assessor.config import RUNS_DIR, SETTINGS
from assessor.graph import Pipeline, new_state
from assessor.tracing import Budget, Tracer

app = FastAPI(title="Nimbu Mart Vendor Risk Assessor", version="0.1.0",
              description="Capstone project by Shivank Agrawal (Future with Shivank). Assessments are inputs to a human decision.")


class AskBody(BaseModel):
    question: str
    user_groups: list[str] = Field(default_factory=lambda: ["public", "employees"])


class AssessBody(BaseModel):
    question: str
    vendor: str
    official_domain: Optional[str] = None
    tier: str = "T3"
    users: int = 1
    user_groups: list[str] = Field(default_factory=lambda: ["public", "employees"])
    requested_by: str = "api"


class ApproveBody(BaseModel):
    decision: str  # approve | reject
    approver: str
    comment: str = ""


@app.get("/health")
def health():
    return {"ok": True, "llm_provider": SETTINGS.llm_provider, "search_provider": SETTINGS.search_provider}


@app.post("/ask")
def ask(body: AskBody):
    pipe = Pipeline(tracer=None)
    s = pipe.run_plain(new_state(body.question, user_groups=body.user_groups, requested_by="api"))
    if s["route"] == "assessment":
        raise HTTPException(400, "This question needs a full assessment: call POST /assess with vendor, tier and users.")
    return {"route": s["route"], "answer": s["answer"],
            "citations": [c.model_dump() for c in (s["policy_answer"].citations if s.get("policy_answer") else [])]}


@app.post("/assess")
def assess(body: AssessBody):
    tracer = Tracer("api")
    pipe = Pipeline(tracer=tracer, budget=Budget())
    s = new_state(body.question, user_groups=body.user_groups, tier=body.tier, users=body.users, vendor=body.vendor,
                  domain=body.official_domain, requested_by=body.requested_by)
    try:
        s = pipe.run(s)
    finally:
        tracer.close()
    if s.get("report") is None:
        return {"route": s["route"], "answer": s["answer"], "run_id": tracer.run_id}
    return {"run_id": tracer.run_id, "report": s["report"].model_dump(mode="json")}


def _run_dir(run_id: str) -> Path:
    p = RUNS_DIR / run_id
    if not p.exists() or ".." in run_id:
        raise HTTPException(404, "run not found")
    return p


@app.post("/approve/{run_id}")
def approve(run_id: str, body: ApproveBody):
    d = _run_dir(run_id)
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    if body.decision == "approve" and rep["overall"] == "BLOCKED":
        raise HTTPException(409, "A BLOCKED assessment cannot be approved. Resolve the blocking requirement or change the data tier.")
    rep["approval"].update({"status": "approved" if body.decision == "approve" else "rejected", "decided_by": body.approver,
                            "decided_on": date.today().isoformat(), "comment": body.comment})
    (d / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(d / "trace.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"stage": "approval", **rep["approval"]}) + "\n")
    return rep["approval"]


@app.get("/runs/{run_id}/report.md", response_class=PlainTextResponse)
def report_md(run_id: str):
    return (_run_dir(run_id) / "report.md").read_text(encoding="utf-8")
