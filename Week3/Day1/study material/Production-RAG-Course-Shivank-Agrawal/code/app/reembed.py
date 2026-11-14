"""
Changing the embedding model without downtime (blue/green re-embedding).

Vectors from two different models live in two different "spaces": you can NEVER compare a query
embedded by model B with chunks embedded by model A. So switching models means re-embedding
everything. Doing it safely:

    1. add a new column   embedding_next vector(<new dims>)
    2. backfill it in batches (resumable: only rows that are still empty)
    3. build its HNSW index
    4. SHADOW-EVALUATE: run the golden questions on old and new, compare hit@5
    5. cut over in one transaction (swap the columns), then restart API + worker with the new EMBED_MODEL
    6. keep the old column for a while so you can roll back, then drop it

Usage:
    python -m app.reembed --embedder fastembed --model BAAI/bge-base-en-v1.5           # steps 1-4
    python -m app.reembed --embedder fastembed --model BAAI/bge-base-en-v1.5 --cutover # + step 5
"""
from __future__ import annotations

import argparse

from rag_core.embeddings import fingerprint, get_embedder
from rag_core.evaluation import is_relevant, load_golden
from rag_core.documents import Chunk

from .db import connect

ap = argparse.ArgumentParser()
ap.add_argument("--embedder", default="fastembed")
ap.add_argument("--model", default=None)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--cutover", action="store_true")
args = ap.parse_args()

old = get_embedder()
new = get_embedder(args.embedder, args.model)
new_id = fingerprint(new)
print(f"current model: {fingerprint(old)}  ->  new model: {new_id}")

with connect() as conn:
    conn.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_next vector({new.dim})")
    conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model_next text")
    conn.commit()

    # 2. backfill in batches (safe to stop and restart)
    total = 0
    while True:
        rows = conn.execute("SELECT id, text FROM chunks WHERE embedding_next IS NULL OR embedding_model_next <> %s "
                            "LIMIT %s", (new_id, args.batch)).fetchall()
        if not rows:
            break
        vecs = new.embed_documents([r["text"] for r in rows])
        with conn.cursor() as cur:
            cur.executemany("UPDATE chunks SET embedding_next = %s, embedding_model_next = %s WHERE id = %s",
                            [(v, new_id, r["id"]) for r, v in zip(rows, vecs)])
        conn.commit()
        total += len(rows)
        print(f"  re-embedded {total} chunks")

    # 3. index for the new column
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_next_hnsw ON chunks "
                 "USING hnsw (embedding_next vector_cosine_ops)")
    conn.commit()

    # 4. shadow evaluation: same golden questions, old vs new vectors (vector search only)
    golden = [g for g in load_golden() if g.get("answerable", True)]
    scores = {"old": 0, "new": 0}
    for g in golden:
        groups = g.get("user_groups", ["public", "employees"])
        for label, emb, col in [("old", old, "embedding"), ("new", new, "embedding_next")]:
            rows = conn.execute(
                f"SELECT c.text, d.filename FROM chunks c JOIN documents d ON d.id = c.document_id "
                f"WHERE c.access && %s ORDER BY c.{col} <=> %s LIMIT 5", (groups, emb.embed_query(g["question"]))
            ).fetchall()
            scores[label] += any(is_relevant(Chunk("x", r["filename"], r["text"]), g) for r in rows)
    n = len(golden)
    print(f"\nshadow eval hit@5: old {scores['old'] / n:.3f}   new {scores['new'] / n:.3f}   (n={n})")

    if args.cutover:
        if scores["new"] < scores["old"]:
            print("New model is WORSE on the golden set - refusing to cut over.")
        else:
            with conn.transaction():  # 5. atomic swap
                conn.execute("ALTER TABLE chunks RENAME COLUMN embedding TO embedding_prev")
                conn.execute("ALTER TABLE chunks RENAME COLUMN embedding_model TO embedding_model_prev")
                conn.execute("ALTER TABLE chunks RENAME COLUMN embedding_next TO embedding")
                conn.execute("ALTER TABLE chunks RENAME COLUMN embedding_model_next TO embedding_model")
                conn.execute("ALTER TABLE chunks ALTER COLUMN embedding_prev DROP NOT NULL")
                conn.execute("ALTER TABLE chunks ALTER COLUMN embedding_model_prev DROP NOT NULL")
            print("Cut over done. Now set EMBEDDER/EMBED_MODEL in .env to the new model and restart the API and worker.")
            print("Roll back = swap the columns back. Drop embedding_prev after a week of healthy traffic.")
