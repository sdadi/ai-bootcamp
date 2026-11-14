"""
policy_rag.py — the internal policy library: Nimbu Mart's own documents, searched with access control.

COURSE MAP
  Production RAG course  Module 6  (chunking: we chunk by clause heading so every hit has a clause id)
                         Module 8  (retrieval: keyword search here; hybrid + reranking is Milestone 1)
                         Module 9  (grounded generation: extractive answers with citations; refuse when weak)
                         Module 11 (security: ACL lives IN retrieval; archived and untrusted docs filtered)
                         Module 13 (the production app, if you go that far)

WHAT IS COMPLETE
  * Loading + clause chunking + metadata (access, status, trust, version)
  * A small pure-Python BM25 so the project runs with no model download
  * ACL / archived / untrusted filters applied BEFORE ranking (never after)
  * Extractive answers with citations and an abstain threshold
  * `approved_vendor_entry()` — reads the restricted Approved Vendor List for PROC-1.1

LEARNER TODO — Milestone 1
  Replace `PolicyLibrary.search()` with the `rag_core` pipeline from the Production RAG course:
  hybrid (BM25 + vectors, RRF) → cross-encoder rerank → threshold. Keep the filters. Then replace
  `answer()`'s extractive mode with grounded LLM generation (rag_core/generate.py): citations required,
  refusal when the context does not contain the answer. Lab 06, Lab 08, Lab 09 are the references.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .config import POLICY_DIR
from .schemas import PolicyAnswer, PolicyHit

STOPWORDS = set("""a an and are as at be by for from has have how if in is it its of on or our that the this to was we what
when where which who will with you your must may can does do did into than then there these those also any all not no
according say says said""".split())

_CLAUSE_RE = re.compile(r"^([A-Z]{1,5}(?:-[\dA-Z]+)+(?:\.\d+)*)\b\s*(.*)$")  # SEC-4.2, AVL-1, Q-SEC-4.2


def tokenize(text: str) -> list[str]:
    text = text.lower()
    # keep clause ids like sec-4.2 as single tokens too
    ids = re.findall(r"\b[a-z]{1,5}-\d+(?:\.\d+)*\b", text)
    words = re.findall(r"[a-z0-9]+", text)
    return [w for w in words if w not in STOPWORDS and len(w) > 1] + ids


@dataclass
class PolicyChunk:
    doc: str
    title: str
    doc_id: str
    clause_id: str
    heading: str
    text: str
    version: str
    status: str
    trust: str
    access: list[str]
    effective_date: str = ""
    tokens: list[str] = field(default_factory=list)


def chunk_markdown(doc: str, text: str, meta: dict) -> list[PolicyChunk]:
    """One chunk per `###` clause (or per `##` section when a section has no clauses).

    Production RAG course Module 6, "Structure-aware chunking": headings are the natural unit of a
    policy. Each chunk keeps its clause id so a citation can say "SEC-4.2", not "chunk 17".
    """
    chunks: list[PolicyChunk] = []
    section = ""
    cur_head, cur_lines, cur_clause = "", [], ""

    def flush():
        body = "\n".join(l for l in cur_lines).strip()
        if body:
            head = cur_head or section or meta.get("title", doc)
            chunks.append(PolicyChunk(
                doc=doc, title=meta.get("title", doc), doc_id=meta.get("doc_id", ""), clause_id=cur_clause,
                heading=head, text=body, version=str(meta.get("version", "")), status=meta.get("status", "active"),
                trust=meta.get("trust", "trusted"), access=list(meta.get("access", ["public"])),
                effective_date=str(meta.get("effective_date", "")),
                tokens=tokenize(head + " " + body)))

    for line in text.splitlines():
        if line.startswith("### "):
            flush()
            cur_lines = []
            raw = line[4:].strip()
            m = _CLAUSE_RE.match(raw)
            cur_clause, cur_head = (m.group(1), raw) if m else ("", raw)
        elif line.startswith("## "):
            flush()
            cur_lines, cur_clause, cur_head = [], "", ""
            section = line[3:].strip()
        elif line.startswith("# "):
            flush()
            cur_lines, cur_clause, cur_head = [], "", ""
        else:
            cur_lines.append(line)
    flush()
    return chunks


class BM25:
    """Minimal BM25 (Production RAG Module 8, "Hybrid search: keywords + meaning" — this is the keyword half)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.n)
        self.df: Counter = Counter()
        for d in docs:
            self.df.update(set(d))
        self.tf = [Counter(d) for d in docs]

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: list[str], idx: int) -> float:
        tf, dl = self.tf[idx], len(self.docs[idx])
        s = 0.0
        for q in query:
            f = tf.get(q, 0)
            if not f:
                continue
            s += self.idf(q) * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


class PolicyLibrary:
    ABSTAIN_BELOW = 2.0  # BM25 score; tune with eval/run_eval.py (Production RAG Module 9 "Refusing: two layers")

    def __init__(self, policy_dir: Path = POLICY_DIR):
        self.policy_dir = Path(policy_dir)
        self.meta: dict = json.loads((self.policy_dir / "_metadata.json").read_text(encoding="utf-8"))
        self.chunks: list[PolicyChunk] = []
        for name, m in self.meta.items():
            if name.startswith("_"):
                continue
            p = self.policy_dir / name
            if not p.exists():
                continue
            self.chunks.extend(chunk_markdown(name, p.read_text(encoding="utf-8"), m))
        self.bm25 = BM25([c.tokens for c in self.chunks])

    # ------------------------------------------------------------------ filters
    @staticmethod
    def _allowed(chunk: PolicyChunk, user_groups: Iterable[str]) -> bool:
        groups = set(user_groups) | {"public"}
        return bool(groups & set(chunk.access))

    def search(self, question: str, user_groups: Iterable[str] = ("public",), *, k: int = 5,
               include_archived: bool = False, trusted_only: bool = True) -> list[PolicyHit]:
        """Filter FIRST (ACL, archived, trust), then rank. Production RAG Module 11: "Access control lives in retrieval"."""
        q = tokenize(question)
        hits: list[PolicyHit] = []
        for i, c in enumerate(self.chunks):
            if not self._allowed(c, user_groups):
                continue
            if c.status == "archived" and not include_archived:
                continue
            if trusted_only and c.trust != "trusted":
                continue
            s = self.bm25.score(q, i)
            if c.clause_id and c.clause_id.lower() in question.lower():
                s += 5.0  # the user named the clause: a hard boost
            if s <= 0:
                continue
            hits.append(PolicyHit(doc=c.doc, title=c.title, doc_id=c.doc_id, clause_id=c.clause_id, heading=c.heading,
                                  text=c.text, score=round(s, 3), version=c.version, status=c.status, trust=c.trust,
                                  access=c.access))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]

    # ------------------------------------------------------------------ answers
    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Split a markdown chunk into citable units: sentences, list items, table rows.

        Policy files are hard-wrapped, so consecutive plain lines are joined into a paragraph first;
        otherwise a sentence that wraps onto the next line would be cut in half.
        """
        parts: list[str] = []
        buf: list[str] = []

        def flush():
            if buf:
                para = " ".join(buf)
                parts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip())
                buf.clear()

        lines = [l.strip() for l in text.splitlines()]
        for i, line in enumerate(lines):
            if not line or set(line) <= set("|-: "):
                flush()
                continue
            if line.startswith("|"):
                flush()
                is_header = i + 1 < len(lines) and lines[i + 1].startswith("|") and set(lines[i + 1]) <= set("|-: ")
                if not is_header:
                    parts.append(line)
            elif line.startswith(("- ", "* ")) or re.match(r"^\d+\.\s", line):
                flush()
                buf.append(line.lstrip("-* "))
            else:
                buf.append(line)
        flush()
        return parts

    def answer(self, question: str, user_groups: Iterable[str] = ("public",), *, llm=None, k: int = 4,
               include_archived: bool = False) -> PolicyAnswer:
        hits = self.search(question, user_groups, k=k, include_archived=include_archived)
        if not hits or hits[0].score < self.ABSTAIN_BELOW:
            return PolicyAnswer(question=question, abstained=True, citations=hits[:2],
                                answer="I could not find this in the policy documents you have access to. "
                                       "If it concerns the Approved Vendor List, ask Procurement.")

        if llm is not None and not llm.mock_mode:
            # LEARNER TODO (Milestone 1): grounded generation with citations — see rag_core/generate.py
            # and Production RAG Module 9. Rules: answer only from the context; cite clause ids; say "not in
            # the policy" when the context does not contain it. Keep this extractive fallback for offline runs.
            pass

        # Extractive mode: the best-matching sentences of the top chunk, prefixed by its clause id.
        q = set(tokenize(question))
        top = hits[0]
        scored = sorted(((len(q & set(tokenize(s))), -i, s) for i, s in enumerate(self._sentences(top.text))), reverse=True)
        best = [s for n, _, s in scored[:2] if n > 0] or [self._sentences(top.text)[0]]
        label = top.clause_id or top.heading
        text = f"{label}: " + " ".join(best) + f"  (source: {top.title}, v{top.version})"
        return PolicyAnswer(question=question, answer=text, citations=hits, mode="extractive")

    # ------------------------------------------------------------------ the Approved Vendor List
    def approved_vendor_entry(self, vendor: str, user_groups: Iterable[str]) -> dict | str | None:
        """Look the vendor up in the restricted Approved Vendor List.

        Returns a dict {status, tier, plan, approved_on, notes}, None if the vendor is not listed,
        or the string "NO_ACCESS" when the caller's groups may not see the list. The matcher turns
        NO_ACCESS into UNKNOWN — the assistant must never leak the list to people without access.
        """
        avl = [c for c in self.chunks if c.doc_id == "NM-PROC-AVL"]
        if not avl:
            return None
        if not any(self._allowed(c, user_groups) for c in avl):
            return "NO_ACCESS"
        v = vendor.lower()
        for c in avl:
            status = {"AVL-1": "approved", "AVL-2": "in_progress", "AVL-3": "rejected"}.get(c.clause_id, "")
            for line in c.text.splitlines():
                if not line.startswith("|") or set(line) <= set("|-: "):
                    continue
                cells = [x.strip() for x in line.strip("|").split("|")]
                if len(cells) < 2 or cells[0].lower() in ("vendor",):
                    continue
                if v in cells[0].lower() or v in cells[1].lower():
                    if status == "approved" and len(cells) >= 7:
                        tier = cells[2].split()[0] if cells[2] else ""
                        return {"status": status, "tier": tier, "plan": cells[3], "approved_on": cells[4],
                                "next_review": cells[5], "notes": cells[6]}
                    if status == "in_progress" and len(cells) >= 5:
                        return {"status": status, "requested_tier": cells[2].split()[0], "notes": cells[4]}
                    if status == "rejected" and len(cells) >= 4:
                        return {"status": status, "date": cells[2], "notes": cells[3]}
        return None
