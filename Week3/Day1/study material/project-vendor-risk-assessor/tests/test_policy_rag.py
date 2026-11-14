"""Policy library: access control, versions, trust — the Production RAG course's Module 11 in test form."""


def test_clauses_are_chunked_with_ids(library):
    ids = {c.clause_id for c in library.chunks}
    assert {"SEC-4.2", "DP-3.1", "AI-2.1", "PROC-1.1", "AVL-1", "Q-SEC-4.2"} <= ids


def test_employee_cannot_see_approved_vendor_list(library):
    hits = library.search("approved vendor list Parakeet", ["public", "employees"])
    assert all(h.doc != "approved_vendor_list.md" for h in hits)


def test_security_can_see_approved_vendor_list(library):
    hits = library.search("Parakeet Translate approved tier", ["public", "employees", "security"])
    assert hits and hits[0].doc == "approved_vendor_list.md"


def test_archived_policy_excluded_by_default(library):
    hits = library.search("SOC 2 Type I customer data", ["public", "employees"])
    assert all(h.status != "archived" for h in hits)
    hits_hist = library.search("SOC 2 Type I customer data", ["public", "employees"], include_archived=True)
    assert any(h.status == "archived" for h in hits_hist)


def test_untrusted_chat_export_never_retrieved(library):
    hits = library.search("LedgerLens approved hackathon channel", ["public", "employees", "security"])
    assert all(h.trust == "trusted" for h in hits)


def test_answer_abstains_on_unrelated_question(library):
    pa = library.answer("What is the best biryani in Bengaluru?", ["public", "employees"])
    assert pa.abstained


def test_answer_cites_clause_and_version(library):
    pa = library.answer("Where must customer personal data be stored?", ["public", "employees"])
    assert not pa.abstained and pa.citations[0].clause_id == "DP-3.1" and "India" in pa.answer and "v2026.1" in pa.answer


def test_approved_vendor_entry_access_and_tiers(library):
    assert library.approved_vendor_entry("Parakeet Translate", ["public", "employees"]) == "NO_ACCESS"
    e = library.approved_vendor_entry("Parakeet Translate", ["security"])
    assert e["status"] == "approved" and e["tier"] == "T2"
    assert library.approved_vendor_entry("LedgerLens AI", ["procurement"])["status"] == "in_progress"
    assert library.approved_vendor_entry("Nobody Inc", ["procurement"]) is None
