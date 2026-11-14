"""
The two data types that flow through the whole pipeline: Document and Chunk.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Page:
    number: int          # 1-based page number (1 for files that have no pages)
    text: str            # extracted text of this page (markdown-ish)
    method: str = "text"  # how we got the text: text | digital | ocr | mixed | table


@dataclass
class Document:
    doc_id: str                      # stable id, here the file name
    text: str                        # full text (all pages joined)
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[Page] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        """SHA-256 of the text. Used for de-duplication and idempotent ingestion."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", self.doc_id))

    def short(self, n: int = 90) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= n else t[: n - 3] + "..."


def stable_chunk_id(doc_id: str, version: str, index: int, text: str) -> str:
    """Deterministic chunk id: the same input always gives the same id.

    This makes re-ingestion idempotent: running ingestion twice upserts the same rows
    instead of creating duplicates.
    """
    h = hashlib.sha1(f"{doc_id}|{version}|{index}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}#{index}-{h}"
