"""
Chapter 8 - Working on the QUERY before retrieval.

    multi_query()   several rephrasings of the question, results fused with RRF
    hyde()          embed a fake answer instead of the question (HyDE)
    rewrite_with_history()  turn "and for laptops?" into a full, stand-alone question
    self_query()    pull structured filters out of the question ("old 2024 policy" -> version=2024)
    route()         decide WHERE to look: documents, a live API, or nowhere

Every function works with an LLM, and has a simple rule-based fallback so the labs run without
an API key. The fallbacks are deliberately simple: they show the idea, not the quality.
"""
from __future__ import annotations

import json
import re

from .llm import LLM

SYNONYMS = {
    "return": ["refund", "send back"], "refund": ["money back", "return"], "phone": ["mobile", "smartphone"],
    "mobile": ["phone"], "price": ["cost"], "cost": ["price", "fee"], "delivery": ["shipping"],
    "shipping": ["delivery"], "leave": ["holiday", "time off"], "warranty": ["guarantee"],
    "cancel": ["stop"], "salary": ["pay", "CTC"], "fee": ["charge", "cost"], "charge": ["fee"],
}


def multi_query(question: str, llm: LLM | None = None, n: int = 3) -> list[str]:
    if llm is not None and llm.available:
        prompt = (f"Write {n} different search queries that would help find documents answering the "
                  f"question below. Vary the wording. Return JSON: {{\"queries\": [...]}}\n\nQuestion: {question}")
        try:
            res = llm.generate("You write search queries.", prompt, json_mode=True)
            queries = [q for q in json.loads(res.text)["queries"] if isinstance(q, str)][:n]
            return [question] + queries
        except Exception:
            pass
    # fallback: synonym expansion
    variants = [question]
    for word, alts in SYNONYMS.items():
        if re.search(rf"\b{word}\b", question, flags=re.I):
            for alt in alts[:1]:
                variants.append(re.sub(rf"\b{word}\b", alt, question, flags=re.I))
    return list(dict.fromkeys(variants))[: n + 1]


def hyde(question: str, llm: LLM | None = None) -> str | None:
    """Hypothetical Document Embeddings: ask the LLM to WRITE a plausible answer passage, then
    search with that passage. Answers look more like documents than questions do.
    Risk: if the fake answer is wrong, it can pull in wrong documents. Always also search
    with the original question."""
    if llm is None or not llm.available:
        return None
    res = llm.generate("You write short policy-style paragraphs.",
                       f"Write a 3-sentence passage from a company policy document that answers:\n{question}")
    return res.text


def rewrite_with_history(history: list[str], question: str, llm: LLM | None = None) -> str:
    """'What about laptops?' only makes sense with the previous turn. Make it stand-alone."""
    if not history:
        return question
    if llm is not None and llm.available:
        convo = "\n".join(f"User: {h}" for h in history[-4:])
        res = llm.generate("Rewrite the last user question so it can be understood without the conversation. "
                           "Return only the rewritten question.", f"{convo}\nUser: {question}")
        return res.text.strip() or question
    # fallback: if the new question is short and vague, glue it to the previous question
    if len(question.split()) <= 5 or re.match(r"(?i)^(and|what about|how about)\b", question):
        return f"{history[-1]} {question}"
    return question


SELF_QUERY_SCHEMA = {
    "department": ["support", "engineering", "hr", "finance", "procurement", "marketing", "catalog", "community"],
    "version": "a year like 2024 or 2025",
    "status": ["active", "archived"],
}


def self_query(question: str, llm: LLM | None = None) -> tuple[str, dict]:
    """Split a question into (search text, metadata filter)."""
    if llm is not None and llm.available:
        prompt = ("Extract metadata filters from the question. Allowed fields and values: "
                  f"{json.dumps(SELF_QUERY_SCHEMA)}. Only use a field if the question clearly asks for it. "
                  'Return JSON: {"query": "<search text>", "filters": {"field": "value"}}\n\n'
                  f"Question: {question}")
        try:
            data = json.loads(llm.generate("You turn questions into search requests.", prompt, json_mode=True).text)
            filters = {k: v for k, v in data.get("filters", {}).items() if k in SELF_QUERY_SCHEMA and v}
            return data.get("query") or question, filters
        except Exception:
            pass
    filters: dict = {}
    year = re.search(r"\b(20\d\d)\b", question)
    if year:
        filters["version"] = year.group(1)
    if re.search(r"\b(old|older|previous|archived|earlier)\b", question, re.I):
        filters["status"] = "archived"
    for words, dept in [(r"leave|salary|payroll|maternity|paternity", "hr"),
                        (r"travel|expense|reimburse|per diem|allowance", "finance"),
                        (r"vendor|seller onboarding|procurement", "procurement"),
                        (r"\bE\d{3}\b|error code|ERR_", "engineering")]:
        if re.search(words, question, re.I):
            filters["department"] = dept
            break
    return question, filters


def route(question: str, llm: LLM | None = None) -> str:
    """Decide where to answer from:
       docs        - policies, FAQs, runbooks (RAG)
       orders_api  - questions about the user's OWN data ("where is my order?") -> call an API, not RAG
       chitchat    - greetings / thanks
       refuse      - out of scope
    """
    if llm is not None and llm.available:
        res = llm.generate("Classify the user's message. Reply with one word: docs, orders_api, chitchat or refuse.\n"
                           "docs = questions about Nimbu Mart policies, products, delivery, payments, HR or internal process.\n"
                           "orders_api = questions about the user's own orders, payments or account.\n"
                           "chitchat = greetings or thanks. refuse = anything else.", question)
        word = res.text.strip().lower().split()[0] if res.text.strip() else "docs"
        return word if word in {"docs", "orders_api", "chitchat", "refuse"} else "docs"
    q = question.lower()
    if re.search(r"\b(my order|my last order|order #?\d+|where is my|my refund|my account)\b", q):
        return "orders_api"
    if re.fullmatch(r"\s*(hi|hello|hey|thanks|thank you|namaste)[!. ]*", q):
        return "chitchat"
    return "docs"
