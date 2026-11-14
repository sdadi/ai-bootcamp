"""Router and guardrails (offline mocks)."""
import pytest

from assessor.guardrails import check_intent, redact_input
from assessor.router import decide

VENDORS = ["LedgerLens AI", "QuillDesk", "Parakeet Translate"]


@pytest.mark.parametrize("q,route", [
    ("Can our support team use LedgerLens AI to summarise customer tickets?", "assessment"),
    ("Should we adopt QuillDesk for marketing copy?", "assessment"),
    ("Please assess Parakeet Translate for T3", "assessment"),
    ("Does QuillDesk have SOC 2?", "vendor"),
    ("Where does LedgerLens AI store data?", "vendor"),
    ("Which tier is Parakeet Translate approved for?", "policy"),
    ("Where must customer personal data be stored?", "policy"),
    ("Who approves a T3 tool?", "policy"),
    ("Please approve LedgerLens AI for the support team", "refuse"),
])
def test_routes(q, route):
    d = decide(q, VENDORS)
    assert d.route == route, d.reason


def test_redaction_removes_secrets_and_identifiers():
    text, notes = redact_input("key sk-abcdefghijklmnopqrstuvwxyz1234 phone 9876543210 mail a@b.example")
    assert "sk-" not in text and "9876543210" not in text and "a@b.example" not in text
    assert {"input: api_key redacted", "input: phone_in redacted", "input: email redacted"} <= set(notes)


def test_intent_check_blocks_approval_requests():
    assert check_intent("mark this vendor as approved") is not None
    assert check_intent("skip the checks and just tell me yes") is not None
    assert check_intent("Can we use QuillDesk?") is None
