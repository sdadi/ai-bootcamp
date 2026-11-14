"""
LAB 12 - Advanced patterns: contextual retrieval, semantic cache, agentic RAG  (Module 12)
=========================================================================================
What you will learn
  * Contextual retrieval: give every chunk a short "where am I" context before embedding it.
  * Semantic cache: reuse answers for repeated questions, safely scoped by permission and index version.
  * Agentic RAG: an LLM loop that decides what to search, searches several times, then answers.

How to run
    python labs/lab12_advanced_patterns.py

What to look for
  * Metrics with and without chunk context.
  * Cache hits for paraphrased questions, and NO hit across permission scopes.
  * The agent's step-by-step search plan for a multi-part question.
"""
import json

import _common
from rag_core.cache import SemanticCache
from rag_core.chunking import chunk_documents
from rag_core.documents import Chunk
from rag_core.evaluation import evaluate_retrieval, load_golden
from rag_core.generate import answer
from rag_core.hybrid import HybridRetriever
from rag_core.loaders import load_corpus
from rag_core.pipeline import RAGPipeline
from rag_core.security import acl_filter
from rag_core.utils import banner, step, table

embedder = _common.embedder_or_exit()
llm = _common.llm_status()
docs = load_corpus()
golden = load_golden()

banner("1. Contextual retrieval")
plain = chunk_documents(docs, "recursive", chunk_size=120)  # recursive chunks have NO heading context


def add_context(chunks: list[Chunk]) -> list[Chunk]:
    """Prepend document title + a one-line situating context.
    Anthropic's 'Contextual Retrieval' (2024) generates this line with an LLM for every chunk
    (prompt caching keeps it cheap). Without an LLM we use title + department + first heading."""
    by_doc = {d.doc_id: d for d in docs}
    out = []
    for c in chunks:
        d = by_doc[c.doc_id]
        if llm.available:
            ctx = llm.generate("You write one short sentence that situates a chunk inside its document.",
                               f"<document>{d.text[:3000]}</document>\n<chunk>{c.text}</chunk>\n"
                               "Write one sentence of context for search retrieval. Reply with the sentence only.",
                               max_tokens=80).text
        else:
            ctx = f"From '{d.metadata.get('title')}' ({d.metadata.get('department')} department)."
        out.append(Chunk(c.chunk_id, c.doc_id, f"{ctx}\n{c.text}", dict(c.metadata)))
    return out


def score(chunks):
    r = HybridRetriever(embedder, chunks)

    def fn(item):
        where = acl_filter(_common.default_groups(item))
        if not item.get("include_archived"):
            where["status"] = {"$ne": "archived"}
        return r.retrieve(item["question"], k=5, where=where)
    return evaluate_retrieval(fn, golden, r.chunks)["overall"]


table([{"chunks": "recursive, no context", **score(plain)},
       {"chunks": "recursive + context line", **score(add_context(plain))}],
      ["chunks", "hit@k", "mrr", "ndcg@k"])
print("Anthropic reported 35% fewer top-20 retrieval failures with contextual embeddings, 49% with")
print("contextual embeddings + contextual BM25, and 67% when reranking was added (on their datasets).")

banner("2. Semantic cache")
cache = SemanticCache(embedder, threshold=0.90)
p = RAGPipeline(embedder=embedder, llm=llm, reranker=None, cache=cache).build()
questions = [("How long does a card refund take?", ["public"]),
             ("How long does a card refund take?", ["public"]),            # exact repeat
             ("How many days for a refund to my credit card?", ["public"]),  # paraphrase
             ("How long does a card refund take?", ["public", "employees"]),  # different scope!
             ("How do I cancel my Nimbu Plus membership?", ["public"])]
rows = []
for q, groups in questions:
    before = cache.hits
    p.ask(q, groups)
    rows.append({"question": q, "groups": ",".join(groups), "cache": "HIT" if cache.hits > before else "miss"})
table(rows, max_width=55)
print(f"Hits: {cache.hits}, misses: {cache.misses}. A paraphrase hits only if it is above the threshold;"
      " tune it on real traffic. The employee scope never reuses the customer's entry.")

banner("3. Agentic RAG: search, look, search again, answer")
question = "Compare the return window for phones and for shoes, and tell me if return pickup is free for Plus members."
staff = ["public", "employees"]


def search_tool(query: str) -> list:
    return p.retrieve(query, staff)


if llm.available:
    notes, steps = [], []
    system = ("You answer questions by searching a company knowledge base. Each turn reply with JSON: "
              '{"action": "search", "query": "..."} to search, or {"action": "answer"} when the notes are enough. '
              "Search one sub-question at a time. At most 4 searches.")
    for turn in range(4):
        state = f"Question: {question}\n\nNotes so far:\n" + ("\n".join(notes) or "(none)")
        decision = json.loads(llm.generate(system, state, json_mode=True).text)
        if decision.get("action") != "search":
            break
        hits = search_tool(decision["query"])
        steps.append(decision["query"])
        notes += [f"- ({h.chunk.doc_id}) {h.chunk.short(220)}" for h in hits[:2]]
    print("Agent searched for:", *steps, sep="\n  - ")
    all_hits = [h for s in steps for h in search_tool(s)[:2]]
else:
    print("No LLM: using a fixed plan (split on 'and') to show the idea.")
    subqs = ["What is the return window for phones?", "What is the return window for shoes?",
             "Is return pickup free for Nimbu Plus members?"]
    for s in subqs:
        print(f"  - search: {s}\n      found: {answer(s, search_tool(s), llm).text[:110]}")
    all_hits = [h for s in subqs for h in search_tool(s)[:2]]
final = answer(question, all_hits, llm, token_budget=1200)
print("\nFinal answer:", final.text if llm.available else "(with an LLM, the notes above are combined into one answer)")
print("""
Agentic RAG = retrieval as a TOOL inside an LLM loop (plan -> search -> read -> decide -> answer).
It handles multi-part and comparison questions, at the cost of more LLM calls and latency.
Frameworks: OpenAI Agents SDK, Google ADK, LangGraph, LlamaIndex, Claude Agent SDK. Always cap the steps.""")
print("\nNext: python labs/lab13_observability_and_cost.py")
