"""
Chapter 11 - Security and privacy for RAG.

1. PII redaction at INGESTION  - personal data never reaches the index, the LLM or the logs.
2. Access control at RETRIEVAL - users can only retrieve chunks their groups may see.
                                 Filter in the search itself, never "ask the LLM to hide it".
3. Prompt-injection defence    - documents can contain instructions written by attackers.
                                 Detect, quarantine, and treat retrieved text as data.

Formats used below (India):
    PAN      5 letters, 4 digits, 1 letter; the 4th letter is the holder type (P = person, C = company ...)
    Aadhaar  12 digits, first digit 2-9, last digit is a Verhoeff checksum
    Mobile   10 digits starting with 6-9, optionally prefixed by +91 or 0
Validating checksums (Verhoeff for Aadhaar, Luhn for cards) avoids redacting random numbers
like order ids.
"""
from __future__ import annotations

import re

# ----------------------------------------------------------------------------- checksums

_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_ok(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def luhn_ok(number: str) -> bool:
    total, alt = 0, False
    for ch in reversed(number):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


# ----------------------------------------------------------------------------- PII

PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), None),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b"), lambda m: verhoeff_ok(re.sub(r"\D", "", m))),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), lambda m: luhn_ok(re.sub(r"\D", "", m))),
    ("PAN", re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]\b"), None),
    ("PHONE", re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{4}\s?\d{5}(?!\d)"), None),
]


def redact_pii(text: str) -> tuple[str, list[dict]]:
    """Replace personal data with placeholders like [PII_PHONE]. Returns (clean_text, findings)."""
    findings = []
    for label, pattern, validator in PII_PATTERNS:
        def _sub(m, label=label, validator=validator):
            value = m.group(0)
            if validator and not validator(value):
                return value  # looks like PII but fails the checksum -> leave it
            findings.append({"type": label, "masked": value[:2] + "***"})
            return f"[PII_{label}]"
        text = pattern.sub(_sub, text)
    return text, findings


# ----------------------------------------------------------------------------- access control


def acl_filter(user_groups: list[str] | tuple[str, ...]) -> dict:
    """Metadata filter: the chunk's 'access' list must share at least one group with the user."""
    return {"access": {"$any": list(user_groups)}}


# ----------------------------------------------------------------------------- prompt injection

INJECTION_PATTERNS = {
    "ignore_instructions": r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|rules|prompts?)\b",
    "fake_system_message": r"\b(system (note|message|prompt)|developer (note|message))\b",
    "role_change": r"\byou are now\b|\bact as\b|\bfrom now on\b",
    "secrecy": r"\bdo not (mention|reveal|tell)\b.{0,30}\b(this|these|note|instructions?)\b",
    "exfiltration": r"\b(reveal|print|show|send)\b.{0,30}\b(system prompt|password|api key|secret|token)\b",
    "markup_injection": r"<\s*/?\s*(script|iframe|system|sources?)\b",
}


def detect_injection(text: str) -> list[str]:
    """Return the names of suspicious patterns found. Heuristics catch the obvious attacks only;
    combine with: trust labels, the 'sources are data' system prompt, least-privilege tools,
    and output checks. A dedicated classifier model is a good next step."""
    t = text.lower()
    return [name for name, pat in INJECTION_PATTERNS.items() if re.search(pat, t, flags=re.S)]


def output_leaks(answer_text: str, banned: list[str]) -> list[str]:
    """Last line of defence: check the final answer for things it must never contain."""
    low = answer_text.lower()
    return [b for b in banned if b.lower() in low]
