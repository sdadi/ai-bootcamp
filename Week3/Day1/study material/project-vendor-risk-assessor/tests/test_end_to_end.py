"""End to end, offline: every gold case must produce the expected verdicts through the whole pipeline.

This is the same check as eval/run_eval.py, in pytest form so it runs in CI with everything else.
"""
import json

import pytest

from assessor.config import GOLD_SET
from assessor.graph import Pipeline, new_state

GOLD = json.loads(GOLD_SET.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pipe():
    return Pipeline(tracer=None)


@pytest.mark.parametrize("case", GOLD["cases"], ids=[c["id"] for c in GOLD["cases"]])
def test_gold_case(pipe, case):
    s = new_state(case["question"], user_groups=case["user_groups"], tier=case["tier"], users=case["users"],
                  vendor=case["vendor"], domain=case.get("official_domain"))
    s = pipe.run_plain(s)
    rep = s["report"]
    got = {r.requirement_id: r.verdict for r in rep.results}
    for rid, exp in case["expected_verdicts"].items():
        assert got[rid] == exp, f"{rid}: expected {exp}, got {got[rid]} — {next(r.reason for r in rep.results if r.requirement_id == rid)}"
    assert rep.overall == case["expected_overall"]
    assert rep.approval.status == "pending"
    for url in case.get("expected_quarantined_urls", []):
        assert url in rep.profile.quarantined
        assert not any(url in r.evidence_urls for r in rep.results)
    for url in case.get("expected_rejected_urls", []):
        assert url in rep.profile.rejected
    # every PASS/FAIL carries evidence (PROC-1.1 cites the policy library instead)
    for r in rep.results:
        if r.verdict in ("PASS", "FAIL") and r.requirement_id != "PROC-1.1":
            assert r.evidence_urls, r.requirement_id
    # the report never claims approval
    md = s["answer"].lower()
    assert "is approved" not in md and "approved for use" not in md


def test_langgraph_engine_matches_plain_engine(pipe):
    pytest.importorskip("langgraph")
    case = GOLD["cases"][0]
    kw = dict(user_groups=case["user_groups"], tier=case["tier"], users=case["users"], vendor=case["vendor"], domain=case.get("official_domain"))
    a = pipe.run_plain(new_state(case["question"], **kw))
    b = pipe.run(new_state(case["question"], **kw), engine="langgraph")
    assert {r.requirement_id: r.verdict for r in a["report"].results} == {r.requirement_id: r.verdict for r in b["report"].results}


def test_blocked_outcome_cannot_be_approved_via_cli_decision(pipe):
    case = GOLD["cases"][0]  # LedgerLens T3 is BLOCKED
    s = new_state(case["question"], user_groups=case["user_groups"], tier="T3", users=35, vendor=case["vendor"], decision="approve", approver="X")
    s = pipe.run_plain(s)
    assert s["report"].overall == "BLOCKED" and s["report"].approval.status == "pending"


def test_policy_and_vendor_routes_return_text(pipe):
    s = pipe.run_plain(new_state("Where must customer personal data be stored?", user_groups=["public", "employees"]))
    assert s["route"] == "policy" and "India" in s["answer"] and s.get("report") is None
    s = pipe.run_plain(new_state("Does QuillDesk have SOC 2?", user_groups=["public", "employees"]))
    assert s["route"] == "vendor" and "SOC 2" in s["answer"]
