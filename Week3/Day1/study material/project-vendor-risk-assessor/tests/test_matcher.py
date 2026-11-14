"""Unit tests for the deterministic matcher. If these pass, the verdict logic is trustworthy; the rest is retrieval."""
from datetime import date

import pytest

from assessor.matcher import Matcher, load_requirements
from assessor.schemas import AssessmentRequest

from conftest import fact, profile


@pytest.fixture(scope="module")
def matcher(library):
    return Matcher(library)


def req(tier="T3", users=35, groups=("public", "employees", "security"), vendor="TestVendor"):
    return AssessmentRequest(question="Can we use it?", vendor=vendor, tier=tier, users=users, user_groups=list(groups))


def verdict(matcher, prof, rid, **kw):
    results = matcher.evaluate(prof, req(**kw))
    return next(r for r in results if r.requirement_id == rid)


def test_requirements_file_loads_and_all_rules_exist():
    reqs = load_requirements()
    assert len(reqs) >= 10
    assert {r.id for r in reqs} >= {"SEC-4.2", "DP-3.1", "AI-2.1", "AI-3.4", "DP-5.4", "SEC-7.1", "SEC-9.3", "DP-2.2", "PROC-1.1"}


# --- SEC-4.2 certification_valid ---------------------------------------------------------------
def test_cert_soc2_type2_recent_documented_passes(matcher):
    p = profile(security_certifications=fact("security_certifications", [{"name": "SOC 2 Type II", "status": "certified", "date": "2026-01-20"}]))
    assert verdict(matcher, p, "SEC-4.2").verdict == "PASS"


def test_cert_soc2_type2_stale_needs_review(matcher):
    p = profile(security_certifications=fact("security_certifications", [{"name": "SOC 2 Type II", "status": "certified", "date": "2024-01-20"}]))
    assert verdict(matcher, p, "SEC-4.2").verdict == "NEEDS_REVIEW"


def test_cert_iso_valid_until_passes_even_if_issued_long_ago(matcher):
    p = profile(security_certifications=fact("security_certifications", [{"name": "ISO 27001", "status": "certified", "date": "2024-01-01", "valid_until": "2027-01-01"}]))
    assert verdict(matcher, p, "SEC-4.2").verdict == "PASS"


def test_cert_claimed_only_is_not_a_pass(matcher):
    p = profile(security_certifications=fact("security_certifications", [{"name": "SOC 2 Type II", "status": "certified", "date": "2026-01-20"}], level="claimed"))
    assert verdict(matcher, p, "SEC-4.2").verdict == "NEEDS_REVIEW"


def test_cert_generic_soc2_needs_review(matcher):
    p = profile(security_certifications=fact("security_certifications", [{"name": "SOC 2", "status": "certified", "date": None}], level="claimed"))
    assert verdict(matcher, p, "SEC-4.2").verdict == "NEEDS_REVIEW"


def test_cert_in_progress_fails(matcher):
    p = profile(security_certifications=fact("security_certifications", [{"name": "ISO 27001", "status": "in_progress", "date": None}]))
    assert verdict(matcher, p, "SEC-4.2").verdict == "FAIL"


def test_cert_missing_is_unknown_with_question(matcher):
    r = verdict(matcher, profile(), "SEC-4.2")
    assert r.verdict == "UNKNOWN" and r.vendor_question_id == "Q-SEC-4.2"


def test_cert_type1_ok_for_t2_but_rule_not_applicable_for_t3(matcher):
    p = profile(security_certifications=fact("security_certifications", [{"name": "SOC 2 Type I", "status": "certified", "date": "2026-03-01"}], level="claimed"))
    assert verdict(matcher, p, "SEC-4.2-T2", tier="T2").verdict == "PASS"
    assert verdict(matcher, p, "SEC-4.2-T2", tier="T3").verdict == "NOT_APPLICABLE"
    assert verdict(matcher, p, "SEC-4.2", tier="T3").verdict == "FAIL"


# --- DP-3.1 region_includes -------------------------------------------------------------------
def test_region_india_available_passes(matcher):
    p = profile(data_regions=fact("data_regions", {"available": ["India", "Singapore"], "planned": []}))
    assert verdict(matcher, p, "DP-3.1").verdict == "PASS"


def test_region_india_planned_fails(matcher):
    p = profile(data_regions=fact("data_regions", {"available": ["United States"], "planned": ["India"]}))
    r = verdict(matcher, p, "DP-3.1")
    assert r.verdict == "FAIL" and "planned" in r.reason


def test_region_unknown_when_not_stated(matcher):
    assert verdict(matcher, profile(), "DP-3.1").verdict == "UNKNOWN"


# --- AI-2.1 enum_verdict ------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [("no", "PASS"), ("no_on_paid_plans", "NEEDS_REVIEW"), ("opt_out_available", "NEEDS_REVIEW"), ("yes", "FAIL")])
def test_training_enum(matcher, value, expected):
    p = profile(trains_on_customer_data=fact("trains_on_customer_data", value))
    assert verdict(matcher, p, "AI-2.1").verdict == expected


# --- AI-3.4 list_disclosed ----------------------------------------------------------------------
def test_subprocessors_with_model_provider_needs_review(matcher):
    p = profile(subprocessors=fact("subprocessors", [{"name": "Orbital", "category": "model_provider", "country": "US"}]))
    r = verdict(matcher, p, "AI-3.4")
    assert r.verdict == "NEEDS_REVIEW" and "Orbital" in r.reason


def test_subprocessors_missing_unknown(matcher):
    assert verdict(matcher, profile(), "AI-3.4").verdict == "UNKNOWN"


# --- DP-5.4 retention_within -------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    ({"default_days": 30, "deletion_on_request_days": 30}, "PASS"),
    ({"default_days": 90, "deletion_on_request_days": 30}, "NEEDS_REVIEW"),
    ({"default_days": 90, "deletion_on_request_days": None}, "NEEDS_REVIEW"),
    ({"default_days": 365, "deletion_on_request_days": 180}, "FAIL"),
])
def test_retention(matcher, value, expected):
    p = profile(data_retention=fact("data_retention", value))
    assert verdict(matcher, p, "DP-5.4").verdict == expected


# --- SEC-7.1 applies_when -----------------------------------------------------------------------
def test_sso_applies_for_many_users_even_on_t2(matcher):
    p = profile(sso_saml=fact("sso_saml", "none"))
    assert verdict(matcher, p, "SEC-7.1", tier="T2", users=50).verdict == "FAIL"
    assert verdict(matcher, p, "SEC-7.1", tier="T2", users=5).verdict == "NOT_APPLICABLE"


# --- SEC-9.3 no_unresolved_incident -----------------------------------------------------------
def test_incident_without_postmortem_fails(matcher):
    p = profile(security_incidents=fact("security_incidents", [{"date": "2026-07-12", "summary": "x", "severity": "major", "postmortem_published": False}]))
    assert verdict(matcher, p, "SEC-9.3").verdict == "FAIL"


def test_incident_with_postmortem_needs_review(matcher):
    p = profile(security_incidents=fact("security_incidents", [{"date": "2025-04-14", "summary": "x", "severity": "major", "postmortem_published": True}]))
    assert verdict(matcher, p, "SEC-9.3").verdict == "NEEDS_REVIEW"


def test_old_incident_outside_window_passes(matcher):
    p = profile(security_incidents=fact("security_incidents", [{"date": "2023-01-01", "summary": "x", "severity": "major", "postmortem_published": False}]))
    assert verdict(matcher, p, "SEC-9.3").verdict == "PASS"


def test_minor_incident_passes(matcher):
    p = profile(security_incidents=fact("security_incidents", [{"date": "2026-08-03", "summary": "x", "severity": "minor", "postmortem_published": True}]))
    r = verdict(matcher, p, "SEC-9.3")
    assert r.verdict == "PASS" and "minor" in r.reason


# --- PROC-1.1 approved_for_tier (reads the policy library) --------------------------------------
def test_avl_hidden_without_access(matcher):
    r = verdict(matcher, profile(), "PROC-1.1", vendor="Parakeet Translate", groups=("public", "employees"))
    assert r.verdict == "UNKNOWN" and "Procurement" in r.reason


def test_avl_t2_approval_not_enough_for_t3(matcher):
    r = verdict(matcher, profile(), "PROC-1.1", vendor="Parakeet Translate", tier="T3")
    assert r.verdict == "NEEDS_REVIEW" and "T2" in r.reason


def test_avl_t2_approval_passes_for_t2(matcher):
    assert verdict(matcher, profile(), "PROC-1.1", vendor="Parakeet Translate", tier="T2").verdict == "PASS"


def test_avl_review_in_progress(matcher):
    r = verdict(matcher, profile(), "PROC-1.1", vendor="LedgerLens AI")
    assert r.verdict == "NEEDS_REVIEW" and "in progress" in r.reason


def test_avl_unlisted_vendor(matcher):
    assert verdict(matcher, profile(), "PROC-1.1", vendor="Nobody Inc").verdict == "NEEDS_REVIEW"


# --- overall -----------------------------------------------------------------------------------------
def test_overall_blocked_on_blocker_fail(matcher):
    p = profile(data_regions=fact("data_regions", {"available": ["United States"], "planned": []}))
    results = matcher.evaluate(p, req())
    overall, reason = Matcher.overall(results)
    assert overall == "BLOCKED" and "DP-3.1" in reason


def test_overall_never_says_approved(matcher):
    results = matcher.evaluate(profile(), req(tier="T1", users=1))
    overall, reason = Matcher.overall(results)
    assert "approved" not in reason.lower() or "approver" in reason.lower()
