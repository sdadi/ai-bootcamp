"""
LAB 09 - Grounded generation: prompts, citations, refusing, structured output  (Module 9)
========================================================================================
What you will learn
  * What the LLM actually receives: system rules + numbered <sources> + the question.
  * Citations you can CHECK: every [S#] must point to a real source; we flag invented ones.
  * Refusing ("I don't have enough information...") is a feature. Two layers make it work:
      1. a similarity threshold BEFORE calling the LLM (cheap, catches nonsense questions)
      2. the prompt rule AFTER (the LLM refuses when the sources do not contain the answer)
  * Context budget and ordering (best evidence at the edges, not lost in the middle).
  * JSON output for apps, and a self-check pass.

How to run
    python labs/lab09_grounded_generation.py

What to look for
  * The similarity histogram: where do answerable and unanswerable questions fall?
    That gap is where you set MIN_SIMILARITY for YOUR embedding model.
"""
import json

import numpy as np

import _common
from rag_core.evaluation import load_golden
from rag_core.generate import answer, build_context, self_check
from rag_core.pipeline import RAGPipeline
from rag_core.prompts import SYSTEM_PROMPT, format_sources, user_prompt
from rag_core.utils import banner, step, table

embedder = _common.embedder_or_exit()
llm = _common.llm_status()
p = RAGPipeline(embedder=embedder, llm=llm, reranker=None).build()
staff = ["public", "employees"]

banner("1. The exact prompt")
q = "How long does a credit card refund take?"
hits = p.retrieve(q, staff)
sources = build_context(hits, token_budget=600)
print("SYSTEM:\n" + SYSTEM_PROMPT)
print("\nUSER:\n" + user_prompt(q, format_sources(sources))[:1600] + "\n...")

banner("2. Answers with checked citations")
for q in ["How long does a credit card refund take?", "Is same-day delivery available in Jaipur?",
          "What is the warranty on a TV display panel?"]:
    a = answer(q, p.retrieve(q, staff), llm)
    print(f"\nQ: {q}\nA ({a.mode}): {a.text}")
    for c in a.citations:
        print(f"   {c['id']} -> {c['title']} (page {c['page']})")
    if a.invalid_citations:
        print("   !! invented citations:", a.invalid_citations)

banner("3. Choosing a refusal threshold from data")
golden = load_golden()
ans_sims, un_sims = [], []
for item in golden:
    if item["type"] == "acl":
        continue
    best = max((h.vector_score or 0) for h in p.retriever.retrieve(item["question"], k=5, mode="vector",
                                                                   where=p.filters(staff, True)))
    (ans_sims if item.get("answerable", True) else un_sims).append(best)
extra = ["Do you deliver to Mars?", "Who won the cricket world cup?", "Write me a poem about mangoes",
         "What is the capital of France?", "How do I bake a cake?"]
for q in extra:
    un_sims.append(max(h.vector_score for h in p.retriever.retrieve(q, k=5, mode="vector", where=p.filters(staff))))
bins = np.linspace(min(ans_sims + un_sims), max(ans_sims + un_sims), 9)
print("best cosine similarity   answerable   unanswerable")
for lo, hi in zip(bins[:-1], bins[1:]):
    a_n = sum(lo <= s < hi or (hi == bins[-1] and s == hi) for s in ans_sims)
    u_n = sum(lo <= s < hi or (hi == bins[-1] and s == hi) for s in un_sims)
    print(f"  {lo:.2f} - {hi:.2f}           {'#' * a_n:<12} {'o' * u_n}")
suggest = (np.percentile(ans_sims, 5) + np.percentile(un_sims, 95)) / 2
print(f"\nA reasonable starting threshold for {embedder.name}: about {suggest:.2f}")
print("Put it in .env as MIN_SIMILARITY=... and re-run Lab 10 to check you did not block real questions.")
print("Note: bge-style models give fairly high scores even to unrelated text, so the threshold is model-specific.")

banner("4. Context budget and 'lost in the middle' ordering")
hits = p.retrieve("What are all the ways to get a refund?", staff)
for budget in [150, 400, 1500]:
    s = build_context(hits, token_budget=budget)
    print(f"budget {budget:5d} tokens -> {len(s)} chunks: {[c.doc_id.split('.')[0] for _, c in s]}")
print("With order='edges', the best chunk is S1 and the second best is LAST (the LLM reads edges best).")

banner("5. Structured (JSON) output and a self-check")
if llm.available:
    a = answer("What is the COD limit?", p.retrieve("What is the COD limit?", staff), llm, json_output=True)
    print("JSON answer text:", a.text)
    print("Self-check:", json.dumps(self_check(a, llm), ensure_ascii=False))
    usage = a.usage
    print(f"tokens in/out: {usage.input_tokens}/{usage.output_tokens}, latency {usage.latency_ms:.0f} ms")
else:
    print("Needs an LLM. With one, you would see a JSON object like:")
    print('  {"answer": "COD is available on orders up to ₹5,000 [S1].", "citations": ["S1"], "answerable": true}')
print("\nNext: python labs/lab10_evaluation.py")
