"""
LAB 07 - Query transformations and routing  (Module 8)
======================================================
What you will learn
  * Multi-query: search with several rephrasings and fuse the results (more recall).
  * HyDE: search with a hypothetical answer instead of the question.
  * Conversation rewriting: "and for shoes?" -> a full question.
  * Routing: "where is my order 12345?" is NOT a RAG question - it needs a live API call
    with the user's own token. RAG is for shared knowledge; APIs are for personal data.

How to run
    python labs/lab07_query_transforms.py

What to look for
  * Without an LLM, multi-query uses a small synonym list (weak but visible effect).
  * With an LLM, compare the generated queries and the HyDE passage.
  * Cost: each extra query is an extra search (and an LLM call to write it). Use these tools
    for the question types that fail in your evaluation, not for everything.
"""
import _common
from rag_core.chunking import chunk_documents
from rag_core.evaluation import evaluate_retrieval, load_golden
from rag_core.hybrid import HybridRetriever
from rag_core.loaders import load_corpus
from rag_core.query_transform import hyde, multi_query, rewrite_with_history, route
from rag_core.security import acl_filter
from rag_core.utils import banner, show_hits, step, table

embedder = _common.embedder_or_exit()
llm = _common.llm_status()
r = HybridRetriever(embedder, chunk_documents(load_corpus(), "structure_aware"))
staff = {"access": {"$any": ["public", "employees"]}, "status": {"$ne": "archived"}}

banner("1. Multi-query")
q = "When do I get my money back for a returned phone?"
queries = multi_query(q, llm)
print("Queries:", *queries, sep="\n  - ")
step("Single query")
show_hits(r.retrieve(q, k=3, where=staff), n=3)
step("Multi-query fused with RRF")
show_hits(r.retrieve_many(queries, k=3, where=staff), n=3)

golden = load_golden()


def fn_factory(use_multi):
    def fn(item):
        where = acl_filter(_common.default_groups(item))
        if not item.get("include_archived"):
            where["status"] = {"$ne": "archived"}
        if use_multi:
            return r.retrieve_many(multi_query(item["question"], llm), k=5, where=where)
        return r.retrieve(item["question"], k=5, where=where)
    return fn


table([{"setup": name, **evaluate_retrieval(fn_factory(m), golden, r.chunks)["overall"]}
       for name, m in [("single query", False), ("multi-query", True)]],
      ["setup", "hit@k", "mrr", "ndcg@k"])

banner("2. HyDE (needs an LLM)")
fake = hyde("What happens if the payment gateway times out?", llm)
if fake:
    print("Hypothetical passage:\n ", fake)
    show_hits(r.retrieve_many(["What happens if the payment gateway times out?", fake], k=3, where=staff), n=3)
else:
    print("Skipped: no LLM configured.")

banner("3. Rewriting a follow-up question")
history = ["What is the return window for phones?"]
follow = "and for shoes?"
standalone = rewrite_with_history(history, follow, llm)
print(f"History: {history}\nFollow-up: {follow!r}\nRewritten: {standalone!r}")
show_hits(r.retrieve(standalone, k=2, where=staff), n=2)

banner("4. Routing: documents vs live data")


def get_order_status(order_id: str, user_token: str) -> dict:
    """Stand-in for a real Orders API. In production this call carries THE USER'S token, so a
    user can only ever see their own orders. Never copy personal order data into a shared index."""
    return {"order_id": order_id, "status": "Out for Delivery", "eta": "today by 10 pm"}


rows = []
for msg in ["What is the COD limit?", "Where is my order 12345?", "Thanks!", "What does E104 mean?"]:
    dest = route(msg, llm)
    action = {"docs": "RAG search", "orders_api": str(get_order_status("12345", "user-token-abc")),
              "chitchat": "small-talk reply", "refuse": "polite refusal"}[dest]
    rows.append({"message": msg, "route": dest, "what happens": action})
table(rows, max_width=70)
print("\nNext: python labs/lab08_reranking.py")
