"""
LAB 10 - An evaluation harness you can run after every change  (Module 10)
=========================================================================
What you will learn
  * Evaluate retrieval and answers separately: you need to know WHICH stage failed.
  * Compare configurations on the same golden set and keep a written report.
  * LLM-as-judge (optional) for faithfulness; RAGAS for the standard metric names.

How to run
    python labs/lab10_evaluation.py              # all configs, retrieval + answers
    python labs/lab10_evaluation.py --judge      # also ask an LLM judge (needs an API key; costs a little)

Output
    reports/eval_report.md   (open it in any Markdown viewer / VS Code)
    reports/eval_report.json

What to look for
  * Which configuration wins, and on which question TYPES it still fails.
  * 'abstain_ok': did the system refuse exactly the questions it should refuse?
"""
import json
import sys
import time

import _common
from rag_core.evaluation import evaluate_answers, evaluate_retrieval, llm_judge, load_golden
from rag_core.pipeline import RAGPipeline
from rag_core.rerank import get_reranker
from rag_core.utils import banner, table

embedder = _common.embedder_or_exit()
llm = _common.llm_status()
reranker = get_reranker()
golden = load_golden()

configs = [
    ("A: vector only, no rerank", dict(mode="vector", reranker=None)),
    ("B: bm25 only, no rerank", dict(mode="bm25", reranker=None)),
    ("C: hybrid (RRF), no rerank", dict(mode="hybrid", reranker=None)),
    ("D: hybrid (RRF) + rerank", dict(mode="hybrid", reranker=reranker)),
]

results = []
for name, cfg in configs:
    banner(name)
    t0 = time.perf_counter()
    p = RAGPipeline(embedder=embedder, llm=llm, **cfg).build()

    def retrieve(item, p=p):
        return p.retrieve(item["question"], _common.default_groups(item), item.get("include_archived", False))

    def ask(item, p=p):
        return p.ask(item["question"], _common.default_groups(item), item.get("include_archived", False))

    ret = evaluate_retrieval(retrieve, golden, p.retriever.chunks, k=5)
    ans = evaluate_answers(ask, golden)
    secs = time.perf_counter() - t0
    results.append({"config": name, "retrieval": ret, "answers": ans, "seconds": round(secs, 1)})
    print({**ret["overall"], **{k: ans[k] for k in ["correct", "abstain_ok", "cited"]}})

banner("Summary")
rows = [{"config": r["config"], "hit@5": r["retrieval"]["overall"]["hit@k"], "MRR": r["retrieval"]["overall"]["mrr"],
         "nDCG@5": r["retrieval"]["overall"]["ndcg@k"], "correct": r["answers"]["correct"],
         "abstain_ok": r["answers"]["abstain_ok"], "cited": r["answers"]["cited"], "secs": r["seconds"]}
        for r in results]
table(rows)

best = max(results, key=lambda r: (r["answers"]["correct"] or 0, r["retrieval"]["overall"]["mrr"]))
fails = [i for i in best["answers"]["items"] if not i["correct"]]
print(f"\nBest config: {best['config']}. Questions it still gets wrong:")
table([{"id": f["id"], "type": f["type"], "answer": f["answer"]} for f in fails], max_width=90)

if "--judge" in sys.argv and llm.available:
    banner("LLM-as-judge on the best config (sample of 8)")
    p = RAGPipeline(embedder=embedder, llm=llm, mode="hybrid", reranker=reranker).build()
    for item in [g for g in golden if g.get("answerable", True)][:8]:
        a = p.ask(item["question"], _common.default_groups(item))
        print(item["id"], llm_judge(llm, item["question"], a.text, [c.text for _, c in a.sources]))
    print("Tip: use a judge from a different provider than the generator, or average several judges.")

s = _common.CODE_DIR / "reports"
s.mkdir(exist_ok=True)
(s / "eval_report.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
md = ["# RAG evaluation report", "", f"Embedding model: `{embedder.name}` | LLM: `{llm}` | Reranker: `{reranker.name}`", "",
      "| Config | hit@5 | MRR | nDCG@5 | correct | abstain_ok | cited |", "|---|---|---|---|---|---|---|"]
md += [f"| {r['config']} | {r['hit@5']} | {r['MRR']} | {r['nDCG@5']} | {r['correct']} | {r['abstain_ok']} | {r['cited']} |"
       for r in rows]
md += ["", "## Retrieval by question type (best config)", "", "| type | hit@5 | MRR | n |", "|---|---|---|---|"]
md += [f"| {t} | {m['hit@k']} | {m['mrr']} | {m['n']} |" for t, m in best["retrieval"]["by_type"].items()]
md += ["", "## Wrong answers (best config)", ""] + [f"- **{f['id']}** ({f['type']}): {f['answer'][:200]}" for f in fails]
(s / "eval_report.md").write_text("\n".join(md), encoding="utf-8")
print(f"\nReport written to {s / 'eval_report.md'}")
print("\nNext: python labs/lab11_security.py")
