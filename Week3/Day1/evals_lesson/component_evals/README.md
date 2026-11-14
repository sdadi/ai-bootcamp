# Component evals and LangGraph: code for the "Evaluating your Agent" lecture

By Shivank Agarwal. Every file uses OpenAI `gpt-6-luna` (and `text-embedding-3-small` for search).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your OpenAI key into .env
```

## Files

| File | What it evaluates | Output |
|---|---|---|
| `langgraph_router.py` | A LangGraph router that sends each question to a reorder, forecast or supplier specialist | Route + answer per question |
| `prompt_evals_inventory.py` | Prompt Version 1 vs Version 2, scored by an LLM judge on Accuracy, Clarity, Completeness | Pass rates + `prompt_eval_results.csv` |
| `rag_evals_inventory.py` | Three RAG setups (k = 1, 2, 3 and three prompts) on Recall@k and three judge metrics | Comparison table + `rag_eval_results.csv` |
| `tool_evals_inventory.py` | A tool-calling agent on 12 scenarios: precision, recall, calls per task, success, latency, cost | Summary + per-task tables |
| `annotator.html` | Human review screen: open in a browser, upload a results CSV, mark 👍 / 👎, download feedback | `..._feedback.csv` |
| `judge_agreement.py` | Does the LLM judge agree with your review? Cohen's kappa + the disagreements | Agreement report |

## A full loop

```bash
python prompt_evals_inventory.py                 # writes prompt_eval_results.csv
# open annotator.html, upload that CSV, review, click "Download feedback"
python judge_agreement.py prompt_eval_results_feedback.csv
```
