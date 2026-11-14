"""
app/ui.py — a one-page Streamlit front end for reviewers.

Run:   streamlit run app/ui.py          (pip install streamlit)

It calls the Pipeline directly (no API) so you can demo without a server. For the team version, call
app/api.py instead and show progress from the trace while a live run is in flight (Milestone 5).

LEARNER TODO — Milestone 5
  * Stream node-by-node progress (route → research → match → …) while a live run takes a minute.
  * Show the trace as a tree next to the report so a reviewer can click from a verdict to the page.
  * Add an "Approve / Reject" form that calls POST /approve/{run_id} with the reviewer's name.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from assessor.config import SETTINGS  # noqa: E402
from assessor.graph import Pipeline, new_state  # noqa: E402
from assessor.tracing import Budget, Tracer  # noqa: E402

st.set_page_config(page_title="Vendor Risk Assessor — Nimbu Mart", layout="wide")
st.title("AI Vendor Risk Assessor")
st.caption(f"Nimbu Mart · LLM: {SETTINGS.llm_provider} · search: {SETTINGS.search_provider} · an input to the review, not an approval")

with st.sidebar:
    st.header("Who is asking")
    groups = st.multiselect("User groups (ACL)", ["public", "employees", "security", "procurement"], default=["public", "employees", "security"])
    tier = st.selectbox("Highest data tier", ["T1", "T2", "T3"], index=2)
    users = st.number_input("Number of users", 1, 10000, 35)
    vendor = st.text_input("Vendor (optional, otherwise detected)", "")
    domain = st.text_input("Official domain (optional)", "")

q = st.text_input("Question", "Can our support team use LedgerLens AI to summarise customer tickets?")
if st.button("Run", type="primary"):
    SETTINGS.verbose = False
    tracer = Tracer("ui")
    pipe = Pipeline(tracer=tracer, budget=Budget())
    with st.spinner("Routing → researching → matching…"):
        s = pipe.run(new_state(q, user_groups=groups, tier=tier, users=int(users), vendor=vendor or None, domain=domain or None, requested_by="ui"))
    tracer.close()
    rep = s.get("report")
    if rep is None:
        st.subheader(f"Route: {s['route']}")
        st.markdown(s["answer"])
    else:
        colour = {"BLOCKED": "red", "CONDITIONAL": "orange", "READY_FOR_APPROVAL": "green"}[rep.overall]
        st.markdown(f"### :{colour}[{rep.overall}] — {rep.overall_reason}")
        rows = [{"Requirement": r.title, "Clause": r.clause, "Verdict": r.verdict, "Severity": r.severity, "Reason": r.reason,
                 "Evidence": ", ".join(r.evidence_urls[:2]), "As of": str(r.evidence_as_of or "")} for r in rep.results]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if rep.open_questions:
            st.subheader("Questions for the vendor")
            for i, oq in enumerate(rep.open_questions, 1):
                st.markdown(f"**{i}. {oq['id']}** — {oq['question']}")
        with st.expander("Full review file (Markdown)"):
            st.markdown(s["answer"])
        with st.expander("Trace"):
            st.json([e for e in tracer.events])
        st.caption(f"Run {rep.run_id} · {rep.llm_calls} LLM calls · est. ${rep.cost_usd:.4f} · {rep.duration_s}s · approvers: {', '.join(rep.approval.approver_roles)}")
