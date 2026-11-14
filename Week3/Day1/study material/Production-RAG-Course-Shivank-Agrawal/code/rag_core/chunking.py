"""
Chapter 6 - Chunking strategies.

All strategies take text and return a list of strings. `chunk_documents()` wraps them,
adds metadata and gives every chunk a stable id.

    fixed_size        cut every N tokens (simple, ignores meaning)
    sliding_window    fixed size + overlap (so facts on a boundary are not lost)
    recursive         split on paragraphs, then lines, then sentences, then words
    structure_aware   split on headings; each chunk carries its heading path
    semantic          split where the topic changes (uses embeddings)
    parent_child      small chunks for searching, big parent chunks for answering
    llm_based         ask an LLM where the topics start (expensive; falls back if no LLM)

Sizes are in TOKENS. We use OpenAI's tiktoken tokenizer if it is available; otherwise we
count words, which is a close-enough estimate for English (1 word is about 1.3 tokens).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from .documents import Chunk, Document, stable_chunk_id

# ----------------------------------------------------------------------------- tokens


class Tokenizer:
    def __init__(self) -> None:
        self.exact = False
        try:
            import tiktoken

            self._enc = tiktoken.get_encoding("cl100k_base")
            self.exact = True
        except Exception:  # offline or not installed -> word-based estimate
            self._enc = None

    def encode(self, text: str) -> list:
        if self._enc is not None:
            return self._enc.encode(text)
        return re.findall(r"\S+\s*", text)  # each "token" is a word plus its trailing space

    def decode(self, tokens: list) -> str:
        if self._enc is not None:
            return self._enc.decode(tokens)
        return "".join(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


TOKENIZER = Tokenizer()


def count_tokens(text: str) -> int:
    return TOKENIZER.count(text)


# ----------------------------------------------------------------------------- strategies


def fixed_size(text: str, chunk_size: int = 120) -> list[str]:
    """Cut the text every `chunk_size` tokens, no matter where sentences end."""
    toks = TOKENIZER.encode(text)
    return [TOKENIZER.decode(toks[i:i + chunk_size]).strip() for i in range(0, len(toks), chunk_size)]


def sliding_window(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Like fixed_size, but each chunk starts `overlap` tokens before the previous one ended."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    toks = TOKENIZER.encode(text)
    step = chunk_size - overlap
    out = []
    for i in range(0, len(toks), step):
        out.append(TOKENIZER.decode(toks[i:i + chunk_size]).strip())
        if i + chunk_size >= len(toks):
            break
    return out


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive(text: str, chunk_size: int = 200, overlap: int = 0,
              separators: list[str] | None = None) -> list[str]:
    """Split on the 'biggest' natural boundary that keeps chunks under chunk_size.

    Try paragraphs first ("\\n\\n"). A paragraph that is still too big is split on lines,
    then sentences, then words. This is the same idea as LangChain's
    RecursiveCharacterTextSplitter (its default separators are "\\n\\n", "\\n", " ", "").
    """
    seps = separators or DEFAULT_SEPARATORS
    pieces = _recursive_split(text, chunk_size, seps)
    pieces = [p.strip() for p in pieces if p.strip()]
    if overlap <= 0:
        return pieces
    out = [pieces[0]] if pieces else []
    for prev, cur in zip(pieces, pieces[1:]):
        tail = TOKENIZER.decode(TOKENIZER.encode(prev)[-overlap:])
        out.append((tail.strip() + " " + cur).strip())
    return out


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    if count_tokens(text) <= size:
        return [text]
    sep_index = next((i for i, s in enumerate(seps) if s == "" or s in text), len(seps) - 1)
    sep = seps[sep_index]
    if sep == "":  # no natural boundary left: hard cut by tokens
        return fixed_size(text, size)
    parts = text.split(sep)
    parts = [p + sep for p in parts[:-1]] + [parts[-1]]  # keep the separator
    chunks: list[str] = []
    current = ""
    for part in parts:
        if count_tokens(part) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_recursive_split(part, size, seps[sep_index + 1:]))
        elif count_tokens(current + part) <= size:
            current += part
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Section:
    path: list[str]   # e.g. ["Returns and Refund Policy (2025)", "4. Refund timelines"]
    body: str


def split_sections(markdown: str) -> list[Section]:
    """Split Markdown into sections by headings, remembering the heading path."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush():
        text = "\n".join(body).strip()
        if text:
            sections.append(Section([h for _, h in stack], text))
        body.clear()

    for line in markdown.split("\n"):
        m = _HEADING.match(line.strip())
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            body.append(line)
    flush()
    return sections


def structure_aware(text: str, max_tokens: int = 250, title: str | None = None,
                    merge_siblings: bool = True) -> list[str]:
    """Chunk along the document's own structure (headings).

    * Every chunk starts with its heading path ("Refund Policy > 4. Refund timelines"), so even a
      short chunk like "Card: 5 to 7 business days" still says WHICH policy and section it is from.
    * Small neighbouring sections under the same parent are merged up to max_tokens
      (otherwise you get many 20-token chunks that are too thin to answer anything).
    * Sections longer than max_tokens are split recursively and keep their heading path.
    * If the document has no top-level heading, the document title is used as the first level.
    """
    sections = split_sections(text)
    if not sections:
        return []
    has_h1 = bool(re.search(r"^#\s", text, flags=re.M))
    groups: list[tuple[list[str], list[str]]] = []  # (parent path, [section texts])
    for sec in sections:
        path = list(sec.path)
        if title and not has_h1:
            path = [title] + path
        parent, name = path[:-1], (path[-1] if path else "")
        part = f"{name}\n{sec.body}" if name and parent else sec.body
        if (merge_siblings and groups and groups[-1][0] == parent and parent
                and count_tokens("\n".join(groups[-1][1] + [part])) <= max_tokens):
            groups[-1][1].append(part)
        else:
            groups.append((parent if parent else path, [part]))
    out: list[str] = []
    for parent, parts in groups:
        header = " > ".join(parent)
        body = "\n\n".join(parts)
        budget = max(40, max_tokens - count_tokens(header))
        for piece in recursive(body, chunk_size=budget):
            out.append(f"{header}\n{piece}" if header else piece)
    return out


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def semantic(text: str, embedder, breakpoint_percentile: float = 80, max_tokens: int = 250,
             min_tokens: int = 25) -> list[str]:
    """Split where the meaning changes.

    1. split into sentences   2. embed every sentence
    3. measure the distance between each sentence and the next
    4. a big jump (above the given percentile of all jumps) = a topic change = a new chunk
    """
    sents = split_sentences(text)
    if len(sents) <= 2:
        return [" ".join(sents)] if sents else []
    vecs = embedder.embed_documents(sents)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)  # vectors are normalised, so dot = cosine
    distances = 1 - sims
    threshold = np.percentile(distances, breakpoint_percentile)
    groups: list[list[str]] = [[sents[0]]]
    for sent, dist in zip(sents[1:], distances):
        if dist > threshold:
            groups.append([sent])
        else:
            groups[-1].append(sent)
    chunks: list[str] = []
    for g in groups:
        joined = " ".join(g)
        if chunks and count_tokens(joined) < min_tokens:
            chunks[-1] += " " + joined  # glue tiny groups to the previous one
        else:
            chunks.extend(recursive(joined, chunk_size=max_tokens))
    return chunks


def llm_based(text: str, llm, max_tokens: int = 250) -> list[str]:
    """Ask an LLM to group paragraphs into topics. Falls back to structure_aware
    when no LLM is configured or the LLM's JSON cannot be read."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if llm is None or not getattr(llm, "available", False) or len(paragraphs) < 3:
        return structure_aware(text, max_tokens)
    numbered = "\n".join(f"[{i}] {p[:300]}" for i, p in enumerate(paragraphs))
    prompt = (
        "Group these numbered paragraphs into chunks so that each chunk covers ONE topic and "
        "can be understood on its own. Keep the original order and use every paragraph once.\n"
        'Reply with JSON only, like {"groups": [[0,1],[2],[3,4]]}.\n\n' + numbered
    )
    try:
        reply = llm.generate("You split documents into coherent chunks.", prompt, json_mode=True)
        groups = json.loads(reply.text)["groups"]
        chunks = [" \n".join(paragraphs[i] for i in g) for g in groups]
        out: list[str] = []
        for c in chunks:
            out.extend(recursive(c, chunk_size=max_tokens))
        return out
    except Exception:
        return structure_aware(text, max_tokens)


# ----------------------------------------------------------------------------- documents -> chunks

# Which document metadata every chunk should carry (used for filters, ACL and citations).
CHUNK_META_KEYS = ["title", "department", "doc_type", "access", "version", "effective_date",
                   "status", "trust", "language", "source_file"]


def _guess_page(doc: Document, chunk_text: str) -> int | None:
    if len(doc.pages) <= 1:
        return doc.pages[0].number if doc.pages else None
    body = chunk_text.split("\n", 1)[-1]
    probe = " ".join(body.split()[:8])
    for p in doc.pages:
        if probe and probe in " ".join(p.text.split()):
            return p.number
    return None


def chunk_documents(docs: list[Document], strategy: str = "structure_aware", embedder=None, llm=None,
                    **params) -> list[Chunk]:
    """Chunk many documents with one strategy and attach metadata to every chunk."""
    chunks: list[Chunk] = []
    for doc in docs:
        title = doc.metadata.get("title")
        if strategy == "fixed_size":
            texts = fixed_size(doc.text, **params)
        elif strategy == "sliding_window":
            texts = sliding_window(doc.text, **params)
        elif strategy == "recursive":
            texts = recursive(doc.text, **params)
        elif strategy == "structure_aware":
            p = dict(params)
            if doc.metadata.get("doc_type") == "table":
                p["merge_siblings"] = False  # keep one table row per chunk
            texts = structure_aware(doc.text, title=title, **p)
        elif strategy == "semantic":
            if embedder is None:
                raise ValueError("semantic chunking needs an embedder")
            texts = semantic(doc.text, embedder, **params)
        elif strategy == "llm_based":
            texts = llm_based(doc.text, llm, **params)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        version = str(doc.metadata.get("version", "1"))
        for i, text in enumerate(t for t in texts if t.strip()):
            meta = {k: doc.metadata[k] for k in CHUNK_META_KEYS if k in doc.metadata}
            meta.update({"chunk_index": i, "strategy": strategy, "n_tokens": count_tokens(text),
                         "page": _guess_page(doc, text)})
            chunks.append(Chunk(stable_chunk_id(doc.doc_id, version, i, text), doc.doc_id, text, meta))
    return chunks


def parent_child(docs: list[Document], parent_tokens: int = 400, child_tokens: int = 80
                 ) -> tuple[list[Chunk], dict[str, Chunk]]:
    """Make big PARENT chunks (by section) and small CHILD chunks inside each parent.

    Search runs over the children (precise matches). When a child matches, we hand its
    parent to the LLM (enough surrounding context to answer well).
    Returns (children, parents_by_id).
    """
    parents = chunk_documents(docs, "structure_aware", max_tokens=parent_tokens)
    parents_by_id = {p.chunk_id: p for p in parents}
    children: list[Chunk] = []
    for p in parents:
        header, _, body = p.text.partition("\n")
        for j, piece in enumerate(recursive(body or p.text, chunk_size=child_tokens)):
            text = f"{header}\n{piece}" if body else piece
            meta = dict(p.metadata, parent_id=p.chunk_id, strategy="parent_child", child_index=j,
                        n_tokens=count_tokens(text))
            children.append(Chunk(f"{p.chunk_id}/c{j}", p.doc_id, text, meta))
    return children, parents_by_id
