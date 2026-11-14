"""
LAB 08 - Two-stage retrieval with a cross-encoder reranker  (Module 8)
=====================================================================
What you will learn
  * Stage 1 (hybrid search) is fast and casts a wide net: 25 candidates.
  * Stage 2 (cross-encoder) reads query + chunk together and re-orders the 25. Keep the top 5.
  * The price: reranking adds latency that grows with the number of candidates.

How to run
    python labs/lab08_reranking.py
    RERANKER_MODEL=BAAI/bge-reranker-v2-m3 python labs/lab08_reranking.py   # stronger, multilingual, bigger

What to look for
  * MRR and nDCG before vs after reranking.
  * Milliseconds per query for reranking 25 candidates.
"""
import time

import _common
from rag_core.chunking import chunk_documents
from rag_core.evaluation import evaluate_retrieval, load_golden
from rag_core.hybrid import HybridRetriever
from rag_core.loaders import load_corpus
from rag_core.rerank import get_reranker
from rag_core.security import acl_filter
from rag_core.utils import banner, show_hits, step, table

embedder = _common.embedder_or_exit()
reranker = get_reranker()
print("Reranker:", reranker.name)
r = HybridRetriever(embedder, chunk_documents(load_corpus(), "structure_aware"))
staff = {"access": {"$any": ["public", "employees"]}, "status": {"$ne": "archived"}}

banner("1. One question, before and after reranking")
q = "Do I have to pay anything to send back a dress I don't like?"
cands = r.retrieve(q, k=25, where=staff, candidates=25)
step("Hybrid top 5 (stage 1)")
show_hits(cands, n=5)
step("After cross-encoder reranking (stage 2)")
show_hits(reranker.rerank(q, cands, top_n=5), n=5)
print(f"Reranking 25 candidates took {reranker.last_latency_ms:.0f} ms")

banner("2. Measured on the golden set")
golden = load_golden()
latencies = []


def fn(rerank):
    def inner(item):
        where = acl_filter(_common.default_groups(item))
        if not item.get("include_archived"):
            where["status"] = {"$ne": "archived"}
        hits = r.retrieve(item["question"], k=25, where=where, candidates=25)
        if rerank:
            t0 = time.perf_counter()
            hits = reranker.rerank(item["question"], hits, top_n=5)
            latencies.append((time.perf_counter() - t0) * 1000)
        return hits[:5]
    return inner


rows = [{"setup": "hybrid only", **evaluate_retrieval(fn(False), golden, r.chunks)["overall"]},
        {"setup": "hybrid + rerank", **evaluate_retrieval(fn(True), golden, r.chunks)["overall"]}]
table(rows, ["setup", "hit@k", "mrr", "ndcg@k", "context_precision"])
if latencies:
    print(f"\nRerank latency: average {sum(latencies) / len(latencies):.0f} ms per query (25 candidates, CPU)")
print("""
Typical production pattern: retrieve 25-100 -> rerank -> keep 5-10.
Hosted rerankers (Cohere Rerank, Voyage rerank, Jina) avoid running the model yourself;
open models (bge-reranker-v2-m3, Qwen3-Reranker, mxbai-rerank) can run on your own GPU/CPU.""")
print("\nNext: python labs/lab09_grounded_generation.py")
