"""
Chapter 10 - Measuring a RAG system.

Retrieval metrics (did we FIND the right chunks?)
    hit@k             1 if at least one relevant chunk is in the top k (many people call this "recall@k")
    recall@k          relevant chunks in top k / all relevant chunks that exist in the index
    precision@k       relevant chunks in top k / k
    MRR               1 / rank of the first relevant chunk (0 if none)
    nDCG@k            rewards relevant chunks near the top, with a log discount
    context_precision RAGAS-style: are the relevant chunks ranked ABOVE the irrelevant ones?

Answer metrics (did we SAY the right thing?)
    correct           the answer contains the key facts from the golden set
    abstain_ok        unanswerable questions were refused, answerable ones were not
    cited             the answer has at least one valid citation and no invented ones
    faithfulness*     share of answer sentences supported by the context
                      (*heuristic word-overlap version here; use an LLM judge or RAGAS in production)

A chunk counts as RELEVANT if it comes from one of the item's relevant_docs AND contains at least
one of its evidence strings. That definition works for every chunking strategy.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .bm25 import tokenize
from .chunking import split_sentences
from .config import ROOT


def load_golden(path: Path | None = None) -> list[dict]:
    path = path or ROOT / "eval" / "golden.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _norm(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", " ".join(text.lower().split()))


def is_relevant(chunk, item: dict) -> bool:
    if chunk.doc_id not in item.get("relevant_docs", []):
        return False
    ev = item.get("evidence", [])
    return not ev or any(_norm(e) in _norm(chunk.text) for e in ev)


# ----------------------------------------------------------------------------- retrieval metrics


def retrieval_metrics(rels: list[bool], total_relevant: int, k: int) -> dict:
    top = rels[:k]
    hits = sum(top)
    first = next((i for i, r in enumerate(top, start=1) if r), None)
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(top, start=1) if r)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k, total_relevant) + 1))
    cp_num = sum((sum(top[:i]) / i) for i, r in enumerate(top, start=1) if r)
    return {
        "hit@k": 1.0 if hits else 0.0,
        "recall@k": hits / total_relevant if total_relevant else 0.0,
        "precision@k": hits / k,
        "mrr": 1 / first if first else 0.0,
        "ndcg@k": dcg / idcg if idcg else 0.0,
        "context_precision": cp_num / hits if hits else 0.0,
    }


def evaluate_retrieval(retrieve_fn, golden: list[dict], all_chunks: list, k: int = 5) -> dict:
    """retrieve_fn(item) -> list of Hit. Skips unanswerable items (nothing to find)."""
    per_item, by_type = [], defaultdict(list)
    for item in golden:
        if not item.get("answerable", True):
            continue
        hits = retrieve_fn(item)
        rels = [is_relevant(h.chunk, item) for h in hits]
        total = sum(is_relevant(c, item) for c in all_chunks)
        m = retrieval_metrics(rels, total, k)
        m["id"], m["type"] = item["id"], item["type"]
        per_item.append(m)
        by_type[item["type"]].append(m)
    return {"overall": _mean(per_item), "by_type": {t: _mean(v) for t, v in by_type.items()}, "items": per_item}


def _mean(rows: list[dict]) -> dict:
    keys = ["hit@k", "recall@k", "precision@k", "mrr", "ndcg@k", "context_precision"]
    return {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys} | {"n": len(rows)} if rows else {}


# ----------------------------------------------------------------------------- answer metrics


def answer_correct(answer_text: str, item: dict) -> bool:
    must = item.get("answer_must_contain", [])
    return all(_norm(m) in _norm(answer_text) for m in must)


def heuristic_faithfulness(answer_text: str, context_texts: list[str], min_overlap: float = 0.6) -> float:
    """Share of answer sentences whose content words mostly appear in the context.
    Cheap and deterministic, but it cannot catch subtle errors - use an LLM judge for that."""
    ctx = set(tokenize(" ".join(context_texts)))
    sents = [re.sub(r"\[S\d+\]", "", s) for s in split_sentences(answer_text)]
    sents = [s for s in sents if tokenize(s)]
    if not sents:
        return 0.0
    ok = sum(len(set(tokenize(s)) & ctx) / len(set(tokenize(s))) >= min_overlap for s in sents)
    return ok / len(sents)


JUDGE_PROMPT = """You grade answers from a RAG system. Given SOURCES, a QUESTION and an ANSWER, return JSON:
{"faithfulness": 0-1 (share of claims supported by the sources),
 "relevance": 0-1 (does it answer the question),
 "reason": "one short sentence"}"""


def llm_judge(llm, question: str, answer_text: str, context_texts: list[str]) -> dict:
    """LLM-as-a-judge. Tip from production: use a judge from a DIFFERENT provider than your
    generator (models tend to prefer their own style), or average 2-3 judges (a 'jury')."""
    srcs = "\n---\n".join(context_texts)
    res = llm.generate(JUDGE_PROMPT, f"SOURCES:\n{srcs}\n\nQUESTION: {question}\n\nANSWER: {answer_text}", json_mode=True)
    try:
        return json.loads(res.text)
    except json.JSONDecodeError:
        return {"faithfulness": None, "relevance": None, "reason": "judge returned invalid JSON"}


def evaluate_answers(ask_fn, golden: list[dict], judge=None) -> dict:
    """ask_fn(item) -> Answer. Returns accuracy-style numbers plus per-item rows."""
    rows = []
    for item in golden:
        a = ask_fn(item)
        ctx = [c.text for _, c in a.sources]
        answerable = item.get("answerable", True)
        row = {
            "id": item["id"], "type": item["type"],
            "correct": (answer_correct(a.text, item) and not a.abstained) if answerable else a.abstained,
            "abstain_ok": a.abstained != answerable,
            "cited": (bool(a.citations) and not a.invalid_citations) if answerable and not a.abstained else None,
            "faithfulness": heuristic_faithfulness(a.text, ctx) if not a.abstained else None,
            "answer": a.text,
        }
        if judge is not None and not a.abstained:
            row["judge"] = llm_judge(judge, item["question"], a.text, ctx)
        rows.append(row)

    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {"correct": rate("correct"), "abstain_ok": rate("abstain_ok"), "cited": rate("cited"),
            "faithfulness_heuristic": rate("faithfulness"), "n": len(rows), "items": rows}
