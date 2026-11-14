"""
LAB 05 - Vector indexes: exact vs HNSW vs IVF vs IVF-PQ, and the filtering trap  (Module 8)
===========================================================================================
What you will learn
  * Exact (brute-force) search is perfect but its cost grows with every vector you add.
  * Approximate nearest neighbour (ANN) indexes trade a little recall for a lot of speed.
  * The knobs that matter: HNSW ef_search, IVF nprobe, PQ compression.
  * The FILTERING TRAP: filtering AFTER an approximate search can return fewer results than you asked for.

How to run
    python labs/lab05_vector_indexes.py               # 50,000 vectors (about 1 minute)
    python labs/lab05_vector_indexes.py --n 200000    # bigger, slower, more realistic

What to look for
  * recall@10 vs query time for each index and setting.
  * In part 2, how many of the 10 requested results survive a 2% filter with post-filtering.
"""
import argparse
import time

import faiss
import numpy as np

import _common  # noqa: F401
from rag_core.utils import banner, table

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50_000)
ap.add_argument("--dim", type=int, default=384)
args = ap.parse_args()
n, d, nq, k = args.n, args.dim, 200, 10

banner(f"Making {n:,} synthetic {d}-dim vectors (clustered, like real embeddings)")
rng = np.random.default_rng(42)
centers = rng.normal(size=(200, d)).astype("float32")
xb = centers[rng.integers(0, 200, n)] + 0.6 * rng.normal(size=(n, d)).astype("float32")
xq = centers[rng.integers(0, 200, nq)] + 0.6 * rng.normal(size=(nq, d)).astype("float32")
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)

flat = faiss.IndexFlatIP(d)  # exact search with inner product (= cosine on normalised vectors)
flat.add(xb)
t0 = time.perf_counter()
_, truth = flat.search(xq, k)
flat_ms = (time.perf_counter() - t0) * 1000 / nq


def recall(found):
    return np.mean([len(set(found[i]) & set(truth[i])) / k for i in range(nq)])


def bench(index, label, knob, values, setter, build_s, mem_mb):
    rows = []
    for v in values:
        setter(v)
        t0 = time.perf_counter()
        _, found = index.search(xq, k)
        ms = (time.perf_counter() - t0) * 1000 / nq
        rows.append({"index": label, "setting": f"{knob}={v}", "build s": round(build_s, 1),
                     "ms/query": round(ms, 3), "recall@10": round(recall(found), 3), "memory MB": mem_mb})
    return rows


rows = [{"index": "Flat (exact)", "setting": "-", "build s": 0.0, "ms/query": round(flat_ms, 3),
         "recall@10": 1.0, "memory MB": round(n * d * 4 / 1e6)}]

banner("1. Building and testing indexes")
t0 = time.perf_counter()
hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)  # M = 32 links per node
hnsw.hnsw.efConstruction = 64
hnsw.add(xb)
b = time.perf_counter() - t0
rows += bench(hnsw, "HNSW (M=32)", "efSearch", [16, 40, 100, 200],
              lambda v: setattr(hnsw.hnsw, "efSearch", v), b, round(n * (d * 4 + 32 * 2 * 4) / 1e6))

nlist = max(16, int(np.sqrt(n)))
t0 = time.perf_counter()
q = faiss.IndexFlatIP(d)
ivf = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
ivf.train(xb)  # k-means to find nlist cluster centres ("Voronoi cells")
ivf.add(xb)
b = time.perf_counter() - t0
rows += bench(ivf, f"IVF-Flat (nlist={nlist})", "nprobe", [1, 8, 32], lambda v: setattr(ivf, "nprobe", v), b,
              round(n * d * 4 / 1e6))

m = 48 if d % 48 == 0 else 32
t0 = time.perf_counter()
q2 = faiss.IndexFlatIP(d)
ivfpq = faiss.IndexIVFPQ(q2, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)  # m sub-vectors x 8 bits
ivfpq.train(xb)
ivfpq.add(xb)
b = time.perf_counter() - t0
rows += bench(ivfpq, f"IVF-PQ (m={m}, 8 bits)", "nprobe", [8, 32], lambda v: setattr(ivfpq, "nprobe", v), b,
              round(n * m / 1e6, 1))
t0 = time.perf_counter()
pq_inner = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
refine = faiss.IndexRefineFlat(pq_inner)  # PQ finds k*10 candidates, exact vectors re-score them
refine.k_factor = 10
refine.train(xb)
refine.add(xb)
b = time.perf_counter() - t0
rows += bench(refine, "IVF-PQ + exact re-rank", "nprobe", [32], lambda v: setattr(pq_inner, "nprobe", v), b,
              f"{round(n * m / 1e6, 1)} + floats")
table(rows)
print("""
Read it like this:
  * HNSW: higher efSearch -> better recall, slower queries. pgvector's default ef_search is 40.
  * IVF: higher nprobe -> more clusters scanned -> better recall, slower.
  * IVF-PQ: ~{:.0f}x less memory than float32, but recall drops; production systems re-rank the
    PQ candidates with full vectors to win it back.
  * At small sizes, exact search is already fast. Do not add an ANN index before you need one.""".format(d * 4 / m))

banner("2. The filtering trap (post-filtering after an ANN search)")
tenant = rng.integers(0, 50, n)          # each vector belongs to one of 50 'tenants' (2% each)
want = 7
hnsw.hnsw.efSearch = 40
_, cand = hnsw.search(xq, 40)            # the ANN index returns only ef_search candidates...
post = [[c for c in row if tenant[c] == want][:k] for row in cand]  # ...then we filter
mask = np.where(tenant == want)[0]
sub = faiss.IndexFlatIP(d)
sub.add(xb[mask])                         # pre-filter: search ONLY inside the allowed vectors
_, pre_idx = sub.search(xq, k)
table([
    {"method": "post-filter (search 40, then keep tenant 7)", "avg results returned (want 10)":
     round(float(np.mean([len(p) for p in post])), 2)},
    {"method": "pre-filter (search only tenant 7)", "avg results returned (want 10)":
     round(float(np.mean([(row >= 0).sum() for row in pre_idx])), 2)},
])
print("""
With a filter matching 2% of rows, an HNSW search that looks at 40 candidates keeps on average
only 40 * 0.02 = 0.8 of them. Users see "no results" even though matching documents exist.
Fixes used in production:
  * pgvector 0.8+: SET hnsw.iterative_scan = relaxed_order;  (keeps scanning until enough rows match)
  * filter-aware engines (Qdrant, Weaviate, Milvus, LanceDB) that apply the filter during graph search
  * partitions / partial indexes per tenant, or exact search when the filter is very selective""")
print("\nNext: python labs/lab06_hybrid_search.py")
