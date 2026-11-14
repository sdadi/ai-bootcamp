"""
extractor.py — turn a fetched page into structured claims.

Live mode: an LLM reads the page (wrapped as untrusted data) and returns a
ClaimList that must pass Pydantic validation.

Mock mode: simple regular expressions do the same job on our fictional corpus,
so you can watch the pipeline without an API key. Regexes are brittle on the
real web — that is why real systems use an LLM here.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from .llm import LLM
from .schemas import FIELD_DESCRIPTIONS, ClaimList, Document, EntityAnchor, ExtractedClaim
from .security import spotlight, strip_hidden

_MONTHS = "January February March April May June July August September October November December".split()
_DATE_RE = re.compile(r"(\d{1,2}) (" + "|".join(_MONTHS) + r") (\d{4})")
_NAME = r"([A-Z][a-z]+ [A-Z][a-z]+)"


def _parse_long_date(text: str) -> date | None:
    m = _DATE_RE.search(text)
    if not m:
        return None
    return datetime.strptime(" ".join(m.groups()), "%d %B %Y").date()


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind("] ", 0, start))
    left = 0 if left == -1 else left + 2
    right = text.find(". ", end)
    right = len(text) if right == -1 else right + 1
    return text[left:right].strip()


def rule_extract(doc: Document) -> ClaimList:
    """Regex-based extraction used in mock mode."""
    text = strip_hidden(doc.text)
    claims: list[ExtractedClaim] = []

    def add(field: str, value: str, m: re.Match, event: date | None = None) -> None:
        claims.append(ExtractedClaim(field=field, value=value, evidence_snippet=_sentence_around(text, m.start(), m.end()),
                                     event_date=event))

    for pat in (r"\bCEO,? " + _NAME, _NAME + r" as (?:new )?CEO", _NAME + r", CEO\b"):
        for m in re.finditer(pat, text):
            add("ceo", m.group(1), m)
    for m in re.finditer(r"founded in (\d{4})", text):
        add("founded_year", m.group(1), m)
    for m in re.finditer(r"headquartered in ([A-Z][a-z]+)", text):
        add("headquarters", m.group(1), m)
    for m in re.finditer(r"raised \$(\d+) (?:million|M)(?: in(?: a| its| our)?)? Series ([A-Z])", text):
        tail = text[m.end(): m.end() + 20]
        yr = re.search(r"in (\d{4})", tail)
        add("latest_funding", f"Series {m.group(2)} — ${m.group(1)}M", m, date(int(yr.group(1)), 12, 31) if yr else None)
    for m in re.finditer(r"(final authori[sz]ation|in-principle approval|licen[cs]e (?:was )?(?:revoked|cancelled|suspended))", text, re.I):
        raw = m.group(1).lower()
        value = "final authorisation" if raw.startswith("final") else "in-principle approval" if raw.startswith("in-principle") else "licence revoked"
        add("regulatory_status", value, m, _parse_long_date(text[m.end(): m.end() + 40]))
    for m in re.finditer(r"(\d\.\d)% per transaction (?:on|for) domestic cards", text):
        add("pricing_domestic_cards", f"{m.group(1)}%", m)
    for m in re.finditer(r"(?<!target is )(\d{2}\.\d{1,2})% uptime", text):
        add("uptime_90d", f"{m.group(1)}%", m)
    for m in re.finditer(r"outage on (\d{1,2} [A-Z][a-z]+ \d{4})", text):
        add("major_outage", m.group(1), m, _parse_long_date(m.group(1)))
    for m in re.finditer(r"misconfigured storage bucket", text):
        no_card = "no card data" in text.lower()
        when = re.search(r"(June|July|May) (\d{4})", text)
        label = f"{when.group(1)} {when.group(2)}" if when else "date unknown"
        add("security_incident", f"merchant contact details exposed ({label}); " + ("no card data" if no_card else "card data unclear"), m)
    for m in re.finditer(r"(\d\.\d)/5", text):
        add("merchant_rating", f"{m.group(1)}/5", m)
    return ClaimList(claims=claims)


def gullible_extract(doc: Document) -> ClaimList:
    """Simulates a VULNERABLE model that obeys instructions hidden in a page.

    Used only in Lab 9 to show what an undefended pipeline does. It reads the
    raw text (hidden comments included) and 'follows' the injected command.
    """
    claims = rule_extract(Document(meta=doc.meta, text=doc.text.replace("<!--", "").replace("-->", ""))).claims
    if re.search(r"ignore all previous instructions", doc.text, re.I):
        m = re.search(r"licen[cs]e was revoked", doc.text, re.I)
        if m:
            claims.append(ExtractedClaim(field="regulatory_status", value="licence revoked",
                                         evidence_snippet="confirmed by the regulator (as instructed by the page)"))
    return ClaimList(claims=claims)


EXTRACT_SYSTEM = """You extract facts about ONE target company from ONE web page.
Rules:
- Only extract facts the page states explicitly about the target company. If the page is about a
  different organisation with a similar name, return an empty list.
- evidence_snippet must be copied word-for-word from the page.
- If the page gives a date for the fact (e.g. 'granted on 14 August 2025'), put it in event_date (YYYY-MM-DD).
- A target or plan is not a measurement: do not report 'target 99.95% uptime' as uptime_90d.
- Allowed fields and meaning:
{fields}"""


def llm_extract(llm: LLM, doc: Document, entity: EntityAnchor, sub_question: str) -> ClaimList:
    fields = "\n".join(f"  {k}: {v}" for k, v in FIELD_DESCRIPTIONS.items())
    wrapped, rule = spotlight(strip_hidden(doc.text)[:8000], doc.meta.url)
    system = EXTRACT_SYSTEM.format(fields=fields) + "\n" + rule
    user = (
        f"Target company: {entity.name} ({entity.city}, {entity.country}); industry: {', '.join(entity.industry_keywords)}.\n"
        f"Look-alikes to ignore: {', '.join(entity.exclude) or 'none known'}.\n"
        f"Research sub-question: {sub_question}\n"
        f"Page title: {doc.meta.title}\nPage published: {doc.meta.publication_date}\n\n{wrapped}"
    )
    return llm.structured(system, user, ClaimList, stage="extract", mock=lambda: rule_extract(doc))
