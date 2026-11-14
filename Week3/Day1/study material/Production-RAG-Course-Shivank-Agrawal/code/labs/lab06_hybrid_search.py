"""
LAB 06 - Hybrid search, score fusion, metadata filters and self-query  (Module 8)
=================================================================================
What you will learn
  * Vector search misses exact identifiers (error codes, SKUs); BM25 catches them.
  * Why "alpha * cosine + beta * BM25" on raw scores is broken, and how min-max scaling or RRF fix it.
  * Metadata filters (department, status, access) and self-query (filters taken from the question).

How to run
    python labs/lab06_hybrid_search.py

What to look for
  * For "E110", compare the top result of vector vs BM25 vs hybrid.
  * The raw score scales: cosine about 0.2-0.9, BM25 often 5-20.
  * The metric table: hybrid should be at least as good as the best single method, especially on
    'keyword' questions.
"""
import _common
from rag_core.chunking import chunk_documents
from rag_core.evaluation import evaluate_retrieval, load_golden
from rag_core.hybrid import HybridRetriever
from rag_core.loaders import load_corpus
from rag_core.query_transform import self_query
from rag_core.security import acl_filter
from rag_core.utils import banner, show_hits, step, table

embedder = _common.embedder_or_exit()
chunks = chunk_documents(load_corpus(), "structure_aware")
r = HybridRetriever(embedder, chunks)
staff = {"access": {"$any": ["public", "employees"]}, "status": {"$ne": "archived"}}

banner("1. Identifiers: vector vs BM25 vs hybrid")
for q in ["What does code E110 mean?", "What is the price of SKU NM-PHN-4821?"]:
    for mode in ["vector", "bm25", "hybrid"]:
        step(f"{mode.upper():7s} | {q}")
        show_hits(r.retrieve(q, k=3, mode=mode, where=staff), n=3)

banner("2. Why raw scores cannot simply be added")
hits = r.retrieve("What does code E110 mean?", k=5, mode="hybrid", where=staff)
table([{"chunk": h.chunk.short(50), "cosine": h.vector_score, "bm25 (raw)": h.bm25_score} for h in hits])
print("BM25 is on a different scale, so 0.5*cosine + 0.5*bm25 is mostly 'BM25 with noise'.")
print("Fix 1: min-max scale both lists to 0..1 first.  Fix 2 (simpler, robust): RRF uses only the ranks.")

banner("3. Measure every mode on the golden set")
golden = load_golden()


def run(mode):
    def fn(item):
        where = acl_filter(_common.default_groups(item))
        if not item.get("include_archived"):
            where["status"] = {"$ne": "archived"}
        return r.retrieve(item["question"], k=5, mode=mode, where=where)
    return evaluate_retrieval(fn, golden, r.chunks, k=5)


rows, by_type = [], []
for mode in ["vector", "bm25", "weighted_naive", "weighted_minmax", "hybrid"]:
    res = run(mode)
    rows.append({"mode": mode, **res["overall"]})
    by_type.append({"mode": mode, **{t: res["by_type"][t]["mrr"] for t in ["keyword", "semantic", "table", "ocr"]
                                     if t in res["by_type"]}})
table(rows, ["mode", "hit@k", "recall@k", "mrr", "ndcg@k", "context_precision"])
print("\nMRR by question type:")
table(by_type)

banner("4. Metadata filters")
q = "How many days of leave do I get?"
step("No filter (employee view)")
show_hits(r.retrieve(q, k=3, where=staff), n=3)
step('Filter {"department": "hr"}')
show_hits(r.retrieve(q, k=3, where={**staff, "department": "hr"}), n=3)

banner("5. Self-query: let the question set the filters")
llm = _common.llm_status()
for q in ["Under the old 2024 policy, how many days did customers have to return electronics?",
          "What is the daily allowance for travel to a Tier-1 city?"]:
    text, filters = self_query(q, llm)
    print(f"\nQ: {q}\n   search text: {text!r}\n   filters: {filters}")
    where = {"access": {"$any": ["public", "employees"]}, **filters}
    show_hits(r.retrieve(text, k=2, where=where), n=2)
print("\nNext: python labs/lab07_query_transforms.py")
