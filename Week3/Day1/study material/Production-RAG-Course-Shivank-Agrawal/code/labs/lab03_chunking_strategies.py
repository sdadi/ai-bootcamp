"""
LAB 03 - Chunking strategies, side by side and MEASURED  (Module 6)
===================================================================
What you will learn
  * What each strategy actually produces on the same document.
  * How to choose: not by opinion, but by retrieval metrics on a golden question set.
  * Why parent-child retrieval gives small chunks for search and big chunks for answers.

How to run
    python labs/lab03_chunking_strategies.py

What to look for
  * fixed_size cuts sentences in half; structure_aware chunks start with their heading path.
  * The table at the end: hit@5 and MRR per strategy (vector search only, so only chunking changes).
"""
import _common
from rag_core.chunking import (chunk_documents, fixed_size, parent_child, recursive, semantic, sliding_window,
                               structure_aware)
from rag_core.evaluation import evaluate_retrieval, load_golden
from rag_core.hybrid import Hit, HybridRetriever
from rag_core.loaders import load_corpus
from rag_core.security import acl_filter
from rag_core.utils import banner, table

docs = load_corpus()
policy = next(d for d in docs if d.doc_id == "refund_policy_2025.md")
embedder = _common.embedder_or_exit()

banner("1. One document, six strategies (first 3 chunks each)")
demo = {
    "fixed_size(80)": fixed_size(policy.text, 80),
    "sliding_window(80, overlap 20)": sliding_window(policy.text, 80, 20),
    "recursive(120)": recursive(policy.text, 120),
    "structure_aware(200)": structure_aware(policy.text, 200, title=policy.metadata["title"]),
    "semantic": semantic(policy.text, embedder),
}
for name, chunks in demo.items():
    print(f"\n### {name}: {len(chunks)} chunks")
    for c in chunks[:3]:
        print("   >", " ".join(c.split())[:150], "...")

banner("2. Measure: which strategy retrieves best on 30 golden questions?")
golden = load_golden()
configs = [
    ("fixed_size", {"chunk_size": 120}),
    ("sliding_window", {"chunk_size": 120, "overlap": 30}),
    ("recursive", {"chunk_size": 200}),
    ("structure_aware", {"max_tokens": 250}),
    ("semantic", {}),
]
rows = []
for strategy, params in configs:
    chunks = chunk_documents(docs, strategy, embedder=embedder, **params)
    r = HybridRetriever(embedder, chunks)

    def fn(item, r=r):
        where = acl_filter(_common.default_groups(item))
        if not item.get("include_archived"):
            where["status"] = {"$ne": "archived"}
        return r.retrieve(item["question"], k=5, mode="vector", where=where)

    m = evaluate_retrieval(fn, golden, r.chunks, k=5)["overall"]
    avg = sum(c.metadata["n_tokens"] for c in chunks) / len(chunks)
    rows.append({"strategy": strategy, "chunks": len(chunks), "avg tokens": round(avg), **m})

# parent-child: search small children, return their parents
children, parents = parent_child(docs, parent_tokens=400, child_tokens=80)
pr = HybridRetriever(embedder, children)


def pc_fn(item):
    where = acl_filter(_common.default_groups(item))
    if not item.get("include_archived"):
        where["status"] = {"$ne": "archived"}
    out, seen = [], set()
    for h in pr.retrieve(item["question"], k=20, mode="vector", where=where):
        pid = h.chunk.metadata["parent_id"]
        if pid not in seen:
            seen.add(pid)
            out.append(Hit(parents[pid], h.score, h.vector_score))
    return out[:5]


m = evaluate_retrieval(pc_fn, golden, list(parents.values()), k=5)["overall"]
rows.append({"strategy": "parent_child (search 80, return 400)", "chunks": len(children),
             "avg tokens": round(sum(c.metadata["n_tokens"] for c in parents.values()) / len(parents)), **m})
table(rows, ["strategy", "chunks", "avg tokens", "hit@k", "mrr", "ndcg@k", "precision@k"])
print("""
How to read this:
  hit@5  = share of questions where a correct chunk is in the top 5 (higher is better)
  MRR    = how high the first correct chunk is (1.0 = always first)
  Bigger chunks raise hit@5 a little but cost more prompt tokens per answer (see 'avg tokens').
  Your numbers depend on the embedding model; the point is to MEASURE, then choose.""")
print("\nNext: python labs/lab04_embeddings.py")
