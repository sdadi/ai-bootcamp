"""
The ingestion job, run by an RQ worker in the background.

    rq worker ingest --with-scheduler --url redis://localhost:6379/0

One job = one document:
    processing -> fetch original from blob store -> extract (OCR if needed) -> clean
    -> redact PII -> chunk -> flag prompt injection -> embed -> REPLACE the document's chunks
    in ONE transaction -> indexed

Why one transaction? If the job crashes half-way, the old chunks stay untouched, and a retry
simply runs the whole job again (idempotent). Failed jobs are retried 3 times with back-off;
after that they stay in RQ's FailedJobRegistry (our "dead-letter queue") and the document
status says 'failed' with the error, so a human can look at it.
"""
from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path

from rag_core.chunking import chunk_documents
from rag_core.embeddings import fingerprint, get_embedder
from rag_core.loaders import load_document
from rag_core.security import detect_injection, redact_pii

from . import settings
from .db import connect
from .storage import get_store

_EMBEDDER = None


def embedder():
    global _EMBEDDER  # load the model once per worker process, not once per job
    if _EMBEDDER is None:
        _EMBEDDER = get_embedder()
    return _EMBEDDER


def set_status(conn, doc_id: str, status: str, error: str | None = None) -> None:
    conn.execute("UPDATE documents SET status = %s, error = %s, updated_at = now() WHERE id = %s",
                 (status, error, doc_id))
    conn.commit()


def ingest_document(doc_id: str) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()
        if row is None:
            return {"skipped": "document was deleted"}
        conn.execute("UPDATE documents SET status = 'processing', attempts = attempts + 1, updated_at = now() "
                     "WHERE id = %s", (doc_id,))
        conn.commit()
        try:
            data = get_store().get(row["blob_key"])
            meta = dict(row["metadata"])
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / row["filename"]
                path.write_bytes(data)
                doc = load_document(path, meta)              # extract + clean (OCR inside)
            doc.doc_id = str(doc_id)
            doc.metadata["access"] = list(row["access"])
            doc.text, pii = redact_pii(doc.text)              # personal data never reaches the index
            chunks = chunk_documents([doc], "structure_aware", max_tokens=settings.CHUNK_MAX_TOKENS)
            if not chunks:
                raise ValueError("no text could be extracted (is it a scan without OCR installed?)")
            emb = embedder()
            vectors = emb.embed_documents([c.text for c in chunks])
            model = fingerprint(emb)
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (id, document_id, chunk_index, text, metadata, access, status, "
                        "injection_flag, embedding, embedding_model) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        [(f"{doc_id}:{i}", doc_id, i, c.text, json.dumps(c.metadata, ensure_ascii=False),
                          list(row["access"]), meta.get("status", "active"), bool(detect_injection(c.text)),
                          v, model) for i, (c, v) in enumerate(zip(chunks, vectors))])
                conn.execute("UPDATE documents SET status = 'indexed', error = NULL, updated_at = now() "
                             "WHERE id = %s", (doc_id,))
            return {"document_id": str(doc_id), "chunks": len(chunks), "pii_redacted": len(pii)}
        except Exception as e:
            conn.rollback()
            set_status(conn, doc_id, "failed", f"{e.__class__.__name__}: {e}"[:500])
            traceback.print_exc()
            raise  # let RQ retry the job
