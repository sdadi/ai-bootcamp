# Production RAG from Scratch — Code

**Course by Shivank Agrawal · Future with Shivank**

This folder has everything you need to run the course labs. Open
`Production_RAG_Course_Shivank_Agrawal.html` (one folder up) in your browser and follow it module by module.
Each module tells you which lab to run here.

## 1. Setup (10 minutes)

You need **Python 3.10 or newer** (3.11 or 3.12 recommended).

```bash
cd code
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
python labs/lab00_check_setup.py
```

**Optional extras**

| What | Why | How |
|---|---|---|
| Tesseract OCR | read scanned PDFs (Lab 02) | macOS `brew install tesseract` · Ubuntu `sudo apt install tesseract-ocr` · Windows: installer from UB-Mannheim (github.com/UB-Mannheim/tesseract/wiki), then add it to PATH |
| An LLM | real generated answers | put `OPENAI_API_KEY=...` **or** `ANTHROPIC_API_KEY=...` in `.env` — or install [Ollama](https://ollama.com), run `ollama pull llama3.2`, set `LLM_PROVIDER=ollama` |
| Docker Desktop | the production app (Module 13) | docker.com |

No API key? Every lab still runs. Answers use a free **extractive** mode that quotes the best matching sentence.

## 2. The labs

| Lab | Module | Command | What you see |
|---|---|---|---|
| 00 | Setup | `python labs/lab00_check_setup.py` | what works and how to fix what doesn't |
| 01 | 2 | `python labs/lab01_rag_in_60_lines.py` | a complete RAG in ~60 lines, and the exact prompt |
| 02 | 5 | `python labs/lab02_extraction_and_ocr.py` | page routing, OCR, tables, cleaning |
| 03 | 6 | `python labs/lab03_chunking_strategies.py` | 6 chunking strategies, measured |
| 04 | 7 | `python labs/lab04_embeddings.py` | similarity, distance metrics, memory, quantization, Hindi |
| 05 | 8 | `python labs/lab05_vector_indexes.py` | exact vs HNSW vs IVF vs PQ; the filtering trap |
| 06 | 8 | `python labs/lab06_hybrid_search.py` | BM25 vs vector vs hybrid (RRF), filters, self-query |
| 07 | 8 | `python labs/lab07_query_transforms.py` | multi-query, HyDE, rewriting, routing |
| 08 | 8 | `python labs/lab08_reranking.py` | cross-encoder reranking, before/after |
| 09 | 9 | `python labs/lab09_grounded_generation.py` | prompt, citations, refusal threshold, JSON |
| 10 | 10 | `python labs/lab10_evaluation.py` | evaluation harness + `reports/eval_report.md` |
| 11 | 11 | `python labs/lab11_security.py` | PII redaction, access control, prompt injection |
| 12 | 12 | `python labs/lab12_advanced_patterns.py` | contextual retrieval, semantic cache, agentic RAG |
| 13 | 12 | `python labs/lab13_observability_and_cost.py` | traces, p50/p95, monthly cost |

Run them all: `python scripts/run_all_labs.py` · Unit tests: `pytest -q`

## 3. The production app (Module 13)

**Option A — everything in Docker (easiest, works on Windows too)**

```bash
docker compose --profile app up --build
# new terminal:
python scripts/ingest_samples.py          # uploads data/raw, waits until indexed
python scripts/ask.py "What does error E102 mean?" --groups public,employees
```

**Option B — databases in Docker, Python on your machine (best for learning/debugging)**

```bash
docker compose up -d db redis
uvicorn app.api:app --reload --port 8000                                   # terminal 1
rq worker ingest --with-scheduler --worker-class rq.worker.SimpleWorker \
   --url redis://localhost:6379/0                                          # terminal 2 (macOS/Linux/WSL)
python scripts/ingest_samples.py                                           # terminal 3
```

Open http://localhost:8000/docs to try every endpoint in the browser.
RQ needs macOS, Linux or WSL2 — on Windows use Option A.

Switch embedding models without downtime: `python -m app.reembed --embedder fastembed --model BAAI/bge-base-en-v1.5`
(add `--cutover` when the shadow evaluation looks good).

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not download model ... huggingface.co` | Your network blocks Hugging Face. Set `EMBEDDER=wordllama` in `.env` (fully offline). The reranker falls back automatically. |
| `tesseract is not installed` warning | Install Tesseract (table above). Without it, the scanned PDF is skipped. |
| LLM error about `temperature` | Leave `LLM_TEMPERATURE` empty; many newer models only accept the default. |
| LLM error `model not found` | Model names change. Check your provider's model list and set `LLM_MODEL`. |
| macOS worker crashes with `objc ... fork()` | Use `--worker-class rq.worker.SimpleWorker` (already in the commands above). |
| ₹ or Hindi shows as `?` on Windows | Run `chcp 65001` once in the terminal, or use Windows Terminal. |

## 5. Folder map

```
code/
  rag_core/      the library (one file per course chapter)
  labs/          runnable lessons 00-13
  app/           production API + worker + Postgres/pgvector + re-embedding
  data/raw/      the Nimbu Mart knowledge base (fictional) + _metadata.json
  eval/          golden.jsonl - 33 test questions with expected answers
  scripts/       sample PDF generator, ingestion, ask, run-all
  tests/         pytest unit tests (no downloads needed)
```

All company names, people and numbers in `data/` are fictional and exist only for teaching.
