"""
Postgres + pgvector: schema, connections and the hybrid search query.

Why Postgres for RAG? If your app already runs on Postgres, pgvector lets you keep vectors next to
metadata, permissions and the rest of your data: one database to back up, transactions for
idempotent re-ingestion, and SQL filters + full-text search in the same query.

Key facts (pgvector 0.8.x):
  * Without an index, pgvector does EXACT search (perfect recall). You create HNSW yourself.
  * HNSW defaults: m=16, ef_construction=64, hnsw.ef_search=40.
  * Filtering with an approximate index happens AFTER the index scan. With ef_search=40 and a filter
    matching 10% of rows you get ~4 rows. pgvector 0.8+ fixes this with iterative index scans:
        SET hnsw.iterative_scan = relaxed_order;
    We turn it on automatically when the server supports it.
  * The 'vector' type can be indexed up to 2,000 dims (halfvec: 4,000).
"""
from __future__ import annotations

import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from rag_core.bm25 import tokenize

from . import settings


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.DATABASE_URL, row_factory=dict_row, autocommit=False)
    register_vector(conn)
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          uuid PRIMARY KEY,
    filename    text NOT NULL,
    sha256      text NOT NULL UNIQUE,          -- same file uploaded twice -> same row (idempotent)
    blob_key    text NOT NULL,
    status      text NOT NULL DEFAULT 'uploaded',  -- uploaded > queued > processing > indexed | failed
    metadata    jsonb NOT NULL DEFAULT '{{}}',
    access      text[] NOT NULL DEFAULT '{{public}}',
    attempts    int NOT NULL DEFAULT 0,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id              text PRIMARY KEY,
    document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     int NOT NULL,
    text            text NOT NULL,
    metadata        jsonb NOT NULL DEFAULT '{{}}',
    access          text[] NOT NULL,           -- copied from the document: filter at search time
    status          text NOT NULL DEFAULT 'active',
    injection_flag  boolean NOT NULL DEFAULT false,
    embedding       vector({dim}) NOT NULL,
    embedding_model text NOT NULL,             -- never mix vectors from two models
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin    ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_access_gin ON chunks USING gin (access);
CREATE INDEX IF NOT EXISTS chunks_document   ON chunks (document_id);
"""


def migrate(dim: int) -> str:
    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(SCHEMA.format(dim=dim))
        conn.commit()
        return pgvector_version(conn)


def pgvector_version(conn) -> str:
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    return row["extversion"] if row else "missing"


def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def tsquery_from_question(question: str) -> str:
    """'What does code E110 mean?' -> 'code | e110 | mean' (OR, so any matching word counts)."""
    words = []
    for tok in tokenize(question):
        words += [w for w in re.split(r"[^a-z0-9]+", tok) if w]
    return " | ".join(dict.fromkeys(words)) or "nothing_to_match"


HYBRID_SQL = """
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(qvec)s) AS rnk
    FROM chunks
    WHERE access && %(groups)s AND NOT injection_flag AND embedding_model = %(model)s
      AND (%(include_archived)s OR status <> 'archived')
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(cand)s
),
kw AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rnk
    FROM chunks, to_tsquery('english', %(tsq)s) AS q
    WHERE tsv @@ q AND access && %(groups)s AND NOT injection_flag
      AND (%(include_archived)s OR status <> 'archived')
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT %(cand)s
)
SELECT c.id, c.document_id, c.text, c.metadata,
       1 - (c.embedding <=> %(qvec)s)                                   AS cosine,
       vec.rnk AS vec_rank, kw.rnk AS kw_rank,
       COALESCE(1.0 / (%(rrf_k)s + vec.rnk), 0) + COALESCE(1.0 / (%(rrf_k)s + kw.rnk), 0) AS rrf
FROM vec FULL OUTER JOIN kw ON vec.id = kw.id
JOIN chunks c ON c.id = COALESCE(vec.id, kw.id)
ORDER BY rrf DESC
LIMIT %(limit)s
"""


def hybrid_search(conn, question: str, qvec: np.ndarray, groups: list[str], model: str,
                  limit: int = 40, include_archived: bool = False, rrf_k: int = 60) -> list[dict]:
    """Vector search + full-text search in ONE SQL query, fused with Reciprocal Rank Fusion."""
    with conn.transaction():
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(settings.HNSW_EF_SEARCH)}")
        if _version_tuple(pgvector_version(conn)) >= (0, 8, 0):
            conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")  # fixes the filtering trap
        return conn.execute(HYBRID_SQL, {
            "qvec": qvec, "groups": list(groups), "model": model, "include_archived": include_archived,
            "tsq": tsquery_from_question(question), "cand": settings.CANDIDATES, "rrf_k": rrf_k, "limit": limit,
        }).fetchall()
