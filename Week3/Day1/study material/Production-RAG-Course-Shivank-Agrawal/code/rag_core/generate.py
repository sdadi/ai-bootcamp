"""
Chapter 9 - From retrieved chunks to a grounded, cited answer.

    build_context()  choose which chunks go into the prompt (de-duplicate, respect a token budget,
                     order them so the best evidence is not lost in the middle)
    answer()         call the LLM (or the extractive fallback), then check the citations
    self_check()     optional second pass: does every claim have support in the sources?
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .bm25 import tokenize
from .chunking import count_tokens, split_sentences
from .hybrid import Hit
from .llm import LLM, LLMResult
from .prompts import ABSTAIN, JSON_INSTRUCTIONS, SELF_CHECK_PROMPT, SYSTEM_PROMPT, format_sources, user_prompt


@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)
    abstained: bool = False
    mode: str = "llm"                  # llm | extractive | abstain
    usage: LLMResult | None = None
    context_tokens: int = 0
    sources: list[tuple[str, object]] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)


def _jaccard(a: str, b: str) -> float:
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / max(1, len(x | y))


def build_context(hits: list[Hit], token_budget: int = 1500, dedupe_threshold: float = 0.85,
                  order: str = "edges") -> list[tuple[str, object]]:
    """Pick chunks for the prompt.

    * de-duplicate: skip a chunk that is almost the same as one already chosen
    * budget: stop adding chunks when the token budget is used up
    * order "edges": put the best chunk FIRST and the second-best LAST. LLMs use information
      at the start and end of a long prompt better than information in the middle
      ("Lost in the Middle", Liu et al. 2023).
    """
    chosen, used = [], 0
    for h in hits:
        if any(_jaccard(h.chunk.text, c.text) >= dedupe_threshold for c in chosen):
            continue
        cost = count_tokens(h.chunk.text)
        if chosen and used + cost > token_budget:
            break
        chosen.append(h.chunk)
        used += cost
    if order == "edges" and len(chosen) > 2:
        chosen = [chosen[0]] + chosen[2:] + [chosen[1]]
    return [(f"S{i}", c) for i, c in enumerate(chosen, start=1)]


def _citations(text: str, sources: list[tuple[str, object]]) -> tuple[list[dict], list[str]]:
    by_id = dict(sources)
    used = list(dict.fromkeys(re.findall(r"\[(S\d+)\]", text)))
    valid, invalid = [], []
    for sid in used:
        chunk = by_id.get(sid)
        if chunk is None:
            invalid.append(sid)  # the model cited a source that does not exist
            continue
        valid.append({"id": sid, "doc_id": chunk.doc_id, "title": chunk.metadata.get("title"),
                      "page": chunk.metadata.get("page"), "chunk_id": chunk.chunk_id})
    return valid, invalid


def _is_abstain(text: str) -> bool:
    t = text.lower()
    return "don't have enough information" in t or "do not have enough information" in t


def _units(body: str) -> list[str]:
    """Split a chunk into answer-sized units: sentences, list items and table rows."""
    units: list[str] = []
    for line in body.split("\n"):
        line = line.strip()
        if not line or re.fullmatch(r"\|?[-| :]+\|?", line):
            continue  # skip empty lines and table separator rows like |---|---|
        units.extend(split_sentences(line) if not line.startswith("|") else [line])
    return units


_FILLER = {"many", "much", "long", "get", "need", "anything", "just", "because", "like", "don", "t", "there",
            "should", "would", "could", "please", "tell", "know", "want", "under", "old", "new", "today"}


def _terms(text: str) -> set[str]:
    """Content words with a very light stemmer, so 'phones' matches 'phone' and 'days' matches 'day'."""
    out = set()
    for t in tokenize(text):
        if t in _FILLER:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.add(t)
    return out


def extractive_answer(question: str, sources: list[tuple[str, object]], min_coverage: float = 0.4) -> str | None:
    """No-LLM fallback: quote the source text that best matches the question.

    * words are weighted by rarity (IDF), so "E110" counts more than "Nimbu"
    * table rows come with their header row; key-value records (CSV rows) come whole
    * if the best unit covers less than `min_coverage` of the question's important words,
      we refuse (return None) instead of quoting something unrelated
    It cannot reason or rephrase, but it is always grounded and it is free.
    """
    import math
    import statistics

    q_terms = _terms(question)
    if not q_terms:
        return None
    # Each candidate unit is scored together with its chunk heading (first line), and table rows
    # together with their header row: "| Annual | ₹999 |" only makes sense under "| Plan | Price |".
    cands = []
    for rank, (sid, chunk) in enumerate(sources):
        heading, _, body = chunk.text.partition("\n")
        body = body or heading
        header_row = next((l.strip() for l in body.split("\n") if l.strip().startswith("|")), None)
        for u in _units(body):
            if u == header_row:
                continue
            context = heading + " " + (header_row or "") if u.startswith("|") else heading
            cands.append((rank, sid, chunk, u, _terms(u), _terms(context)))
    if not cands:
        return None
    df: dict[str, int] = {}
    for c in cands:
        for t in c[4]:
            df[t] = df.get(t, 0) + 1
    n = len(cands)
    seen = {t: math.log(1 + n / df[t]) for t in q_terms if t in df}
    fallback = statistics.median(seen.values()) if seen else 1.0  # weight for words no source uses
    idf = {t: seen.get(t, fallback) for t in q_terms}
    total = sum(idf.values())

    best = None
    for rank, sid, chunk, u, u_terms, ctx_terms in cands:
        own = sum(idf[t] for t in q_terms & u_terms)
        with_ctx = sum(idf[t] for t in q_terms & (u_terms | ctx_terms))
        if len(u.split()) < 6 and not re.search(r"\d", u):
            own *= 0.5  # short lines without numbers are usually headings, not answers
        score = own + 0.8 * with_ctx - 0.03 * rank * total
        if best is None or score > best[0]:
            best = (score, sid, chunk, u, with_ctx / total)
    _, sid, chunk, unit, coverage = best
    if coverage < min_coverage:
        return None  # nothing in the sources really matches: refuse instead of quoting noise

    body = [l.strip() for l in chunk.text.split("\n")[1:] if l.strip()]
    if unit.startswith("|"):  # a table row: show the header row too
        header = next((l for l in body if l.startswith("|")), "")
        text = f"{header}\n{unit}" if header and header != unit else unit
    elif sum(1 for l in body if re.match(r"^- [\w ]+: ", l)) >= 3:  # a key-value record (CSV row)
        text = "; ".join(l[2:] for l in body if l.startswith("- "))
    else:
        text = unit
    return f"{text} [{sid}]"


def answer(question: str, hits: list[Hit], llm: LLM | None = None, token_budget: int = 1500,
           min_similarity: float | None = None, json_output: bool = False) -> Answer:
    if not hits:
        return Answer(ABSTAIN, abstained=True, mode="abstain")
    best_sim = max((h.vector_score or -1.0) for h in hits)
    if min_similarity is not None and best_sim < min_similarity:
        # retrieval found nothing close enough: refusing is better than guessing
        return Answer(ABSTAIN, abstained=True, mode="abstain")

    sources = build_context(hits, token_budget=token_budget)
    ctx_tokens = sum(count_tokens(c.text) for _, c in sources)

    if llm is None or not llm.available:
        text = extractive_answer(question, sources)
        if text is None:
            return Answer(ABSTAIN, abstained=True, mode="abstain", sources=sources, context_tokens=ctx_tokens)
        cites, invalid = _citations(text, sources)
        return Answer(text, cites, False, "extractive", None, ctx_tokens, sources, invalid)

    prompt = user_prompt(question, format_sources(sources))
    if json_output:
        prompt += "\n\n" + JSON_INSTRUCTIONS
    result = llm.generate(SYSTEM_PROMPT, prompt, json_mode=json_output)
    text = result.text
    if json_output:
        try:
            data = json.loads(text)
            text = data.get("answer", "")
        except json.JSONDecodeError:
            pass  # keep the raw text; a production system would retry once
    cites, invalid = _citations(text, sources)
    return Answer(text, cites, _is_abstain(text), "llm", result, ctx_tokens, sources, invalid)


def self_check(answer_obj: Answer, llm: LLM) -> dict:
    """Ask the LLM to verify its own answer against the same sources (Chapter 9, self-correction)."""
    if not llm or not llm.available or answer_obj.abstained:
        return {"supported": None, "unsupported_claims": []}
    prompt = f"{format_sources(answer_obj.sources)}\n\nANSWER:\n{answer_obj.text}"
    res = llm.generate(SELF_CHECK_PROMPT, prompt, json_mode=True)
    try:
        return json.loads(res.text)
    except json.JSONDecodeError:
        return {"supported": None, "unsupported_claims": [], "raw": res.text}
