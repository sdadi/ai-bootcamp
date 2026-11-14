"""
The RAG HTTP API.

    uvicorn app.api:app --reload --port 8000
    open http://localhost:8000/docs   (interactive API docs, made by FastAPI)

Endpoints
    POST   /documents        upload a file (+ metadata) -> 202 Accepted, ingestion runs in the background
    GET    /documents        list documents and their status
    GET    /documents/{id}   status of one document: uploaded/queued/processing/indexed/failed
    DELETE /documents/{id}   remove a document and all its chunks
    POST   /query            ask a question -> answer + citations + trace id + timings
    GET    /health           database, queue and model checks

About user groups: here the caller sends them in the X-User-Groups header to keep the demo simple.
In a real system NEVER trust that header from a browser: your auth gateway (or a verified JWT)
must set the user's groups. The search query then filters by them.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue, Retry

from rag_core.documents import Chunk
from rag_core.embeddings import fingerprint, get_embedder
from rag_core.generate import answer
from rag_core.hybrid import Hit
from rag_core.llm import LLM
from rag_core.observability import Tracer
from rag_core.rerank import get_reranker

from . import settings
from .db import connect, hybrid_search, migrate, pgvector_version
from .storage import get_store
from .worker import ingest_document

app = FastAPI(title="Nimbu Mart RAG API", version="1.0",
              description="Production RAG reference app - Future with Shivank (Shivank Agrawal)")

EMBEDDER = get_embedder()
MODEL_ID = fingerprint(EMBEDDER)
LLM_CLIENT = LLM()
RERANKER = get_reranker() if settings.USE_RERANKER else None
TRACER = Tracer(Path(settings.BLOB_DIR).parent / "logs" / "api_traces.jsonl")
QUEUE = Queue(settings.QUEUE_NAME, connection=Redis.from_url(settings.REDIS_URL))
PGVECTOR_VERSION = migrate(EMBEDDER.dim)  # create tables on start-up (use Alembic in a bigger app)


def _groups(header: str | None) -> list[str]:
    groups = [g.strip() for g in (header or "public").split(",") if g.strip()]
    return groups or ["public"]


def _enqueue(doc_id: str) -> None:
    # 3 retries after 10 s, 30 s and 60 s. Needs `rq worker --with-scheduler`.
    QUEUE.enqueue(ingest_document, doc_id, job_timeout=600, retry=Retry(max=3, interval=[10, 30, 60]))


@app.post("/documents", status_code=202)
async def upload(file: UploadFile = File(...), title: str = Form(""), department: str = Form("general"),
                 access: str = Form("public"), doc_type: str = Form("document"), version: str = Form("1"),
                 status: str = Form("active"), trust: str = Form("trusted")):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"file type {ext or '?'} not supported; allowed: {sorted(settings.ALLOWED_EXTENSIONS)}")
    data = await file.read()
    if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file is larger than {settings.MAX_UPLOAD_MB} MB")
    if not data:
        raise HTTPException(400, "empty file")
    sha = hashlib.sha256(data).hexdigest()
    with connect() as conn:
        existing = conn.execute("SELECT id, status FROM documents WHERE sha256 = %s", (sha,)).fetchone()
        if existing:  # idempotent: the same bytes were uploaded before
            if existing["status"] == "failed":
                _enqueue(str(existing["id"]))
            return {"id": str(existing["id"]), "status": existing["status"], "duplicate": True}
        doc_id = str(uuid.uuid4())
        key = f"documents/{doc_id}/{file.filename}"
        get_store().put(key, data)
        meta = {"title": title or Path(file.filename).stem.replace("_", " ").title(), "department": department,
                "doc_type": doc_type, "version": version, "status": status, "trust": trust}
        conn.execute("INSERT INTO documents (id, filename, sha256, blob_key, status, metadata, access) "
                     "VALUES (%s, %s, %s, %s, 'queued', %s, %s)",
                     (doc_id, file.filename, sha, key, json.dumps(meta), [a.strip() for a in access.split(",")]))
        conn.commit()
    _enqueue(doc_id)
    return {"id": doc_id, "status": "queued", "duplicate": False}


@app.get("/documents")
def list_documents():
    with connect() as conn:
        rows = conn.execute("SELECT d.id, d.filename, d.status, d.access, d.attempts, d.error, "
                            "(SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunks "
                            "FROM documents d ORDER BY d.created_at").fetchall()
    return [{**r, "id": str(r["id"])} for r in rows]


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    with connect() as conn:
        r = conn.execute("SELECT id, filename, status, attempts, error, metadata, access, updated_at, "
                         "(SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunks "
                         "FROM documents d WHERE id = %s", (doc_id,)).fetchone()
    if not r:
        raise HTTPException(404, "document not found")
    return {**r, "id": str(r["id"])}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    with connect() as conn:
        r = conn.execute("DELETE FROM documents WHERE id = %s RETURNING blob_key", (doc_id,)).fetchone()
        conn.commit()
    if not r:
        raise HTTPException(404, "document not found")
    get_store().delete(r["blob_key"])  # chunks were deleted by ON DELETE CASCADE
    return {"deleted": doc_id}


class Question(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    include_archived: bool = False
    top_k: int = Field(settings.TOP_K, ge=1, le=20)


@app.post("/query")
def query(body: Question, x_user_groups: str | None = Header(default=None)):
    groups = _groups(x_user_groups)
    trace = TRACER.start("query", groups=groups, question=body.question)
    with trace.span("embed_query"):
        qvec = EMBEDDER.embed_query(body.question)
    with trace.span("hybrid_search") as sp, connect() as conn:
        rows = hybrid_search(conn, body.question, qvec, groups, MODEL_ID, limit=settings.CANDIDATES,
                             include_archived=body.include_archived)
        sp["candidates"] = len(rows)
    hits = [Hit(Chunk(r["id"], str(r["document_id"]), r["text"], r["metadata"]), float(r["rrf"]), float(r["cosine"]))
            for r in rows]
    if RERANKER and hits:
        with trace.span("rerank", model=RERANKER.name):
            hits = RERANKER.rerank(body.question, hits, top_n=body.top_k)
    hits = hits[: body.top_k]
    with trace.span("generate") as sp:
        result = answer(body.question, hits, LLM_CLIENT, min_similarity=LLM_CLIENT.s.min_similarity)
        sp.update(mode=result.mode, abstained=result.abstained)
        if result.usage:
            sp.update(input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens)
    done = trace.end()
    return {"answer": result.text, "abstained": result.abstained, "mode": result.mode,
            "citations": result.citations, "trace_id": done["trace_id"],
            "timings_ms": {s["name"]: s["ms"] for s in done["spans"]}, "total_ms": done["total_ms"]}


@app.get("/health")
def health():
    status = {"embedding_model": MODEL_ID, "llm": repr(LLM_CLIENT),
              "reranker": RERANKER.name if RERANKER else None}
    try:
        with connect() as conn:
            status["pgvector"] = pgvector_version(conn)
            status["indexed_chunks"] = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    except Exception as e:
        raise HTTPException(503, f"database unavailable: {e}")
    try:
        status["queue_waiting"] = len(QUEUE)
    except Exception as e:
        raise HTTPException(503, f"redis unavailable: {e}")
    return status


@app.get("/")
def root():
    return {"message": "Nimbu Mart RAG API. Open /docs to try it.", "started": time.ctime()}
