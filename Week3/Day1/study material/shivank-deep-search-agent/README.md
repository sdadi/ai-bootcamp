# Deep Search Agents — Course Code

**Build a Multi-Agent Deep Search System** · by **Shivank Agrawal** · *Future with Shivank*

This folder is the hands-on half of the course. The other half is the course page,
`Deep-Search-Agents-Shivank-Agrawal.html`, which you open in your browser. Read a module there, then run its lab here.

Everything runs **offline by default**: no API key, no internet, no cost. When you are ready,
you switch on a real LLM (several free options) and real web search by editing one file.

---

## 1. What you will build

A research agent that answers *"Is NimbusPay a reliable payment-gateway partner for our startup today?"*
by **planning → searching in parallel → validating → merging → evaluating → reporting**.

```
question ─► PLAN (create → review)
              │
              ▼
        ┌─► EXECUTE  parallel workers: search → fetch → injection scan → extract claims
        │     ▼
        │   VALIDATE identity · authority · freshness · copies · quarantine
        │     ▼
        │   MERGE    dedupe → resolve conflicts → keep provenance
        │     ▼
        └── EVALUATE continue? ──stop──► SYNTHESISE report.md + report.json + trace.jsonl
```

NimbusPay and every page in `data/corpus.json` are **fictional**. The small fake web is deliberately messy, with:
- old pages
- a copied press release
- a same-name UK company
- a rumour blog
- a page carrying a **prompt-injection attack**

---

## 2. Setup (10 minutes)

You need **Python 3.10 or newer**. Check with `python3 --version` (macOS/Linux) or `python --version` (Windows).
If you don't have it, install it from python.org.

### macOS / Linux
```bash
cd shivank-deep-search-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run_agent.py
```

### Windows (PowerShell)
```powershell
cd shivank-deep-search-agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python run_agent.py
```
If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**It worked if** you see a report with these values:
- CEO **Ananya Rao**
- regulatory status **final authorisation**
- latest funding **Series B — $40M**

It should also show a **Security notes** section that quarantined one page.

Run the automated checks with:
```bash
python -m pytest -q
```
All 16 should pass.

---

## 3. The labs — run them in this order

Always run labs **from the project root folder** (the folder that contains this README).

| Lab | Command | Course module | What you learn |
|---|---|---|---|
| 1 | `python labs/lab01_search_vs_rag_vs_deep.py` | 1 | Same question, three systems, scored against gold answers |
| 2 | `python labs/lab02_tools_and_schemas.py` | 4 | Tool descriptions, Pydantic validation, repair loop, tool-calling loop |
| 3 | `python labs/lab03_planner.py` | 3 | Plan Creator → Reviewer → Evaluator, stop rules |
| 4 | `python labs/lab04_executor_parallel.py` | 4, 9 | Workers, evidence objects, parallel speed-up |
| 5 | `python labs/lab05_mcp_client.py` | 4 | Build an MCP server and call it from a client |
| 6 | `python labs/lab06_validation.py` | 5 | Reliability scoring, identity, copies, four dates, freshness decay |
| 7 | `python labs/lab07_merge_and_conflicts.py` | 6 | Dedup thresholds, conflict resolution, provenance, final output |
| 8 | `python labs/lab08_iteration_memory_context.py` | 7 | Iterative loop, context compaction, long-term memory |
| 9 | `python labs/lab09_prompt_injection.py` | 8 | Attack the agent, then defend it in layers |
| 10 | `python labs/lab10_cost_latency_budget.py` | 9 | Token costs, model routing, hard budgets, latency |
| 11 | `python labs/lab11_evaluation.py` | 10 | Gold set, metrics, ablation, LLM-as-judge |
| 12 | `python labs/lab12_langgraph_version.py` | 2, 12 | The same agent as a LangGraph state machine |
| ★ | `python run_agent.py` | 12 | Capstone: the full agent from the command line |

After any agent run, open the newest folder in `runs/`. It contains four files:
- `report.md` — the human report
- `report.json` — the same report, structured
- `evidence.json` — every claim with its scores
- `trace.jsonl` — every step the agent took

Reading the trace is the fastest way to understand the agent.

---

## 4. Switch on a real LLM (optional, recommended after Lab 4)

Open `.env` and uncomment **one** provider block.

| Provider | Cost | What to put in `.env` |
|---|---|---|
| **Gemini** | free tier | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...` (key: aistudio.google.com/apikey) |
| **Groq** | free plan with rate limits | `LLM_PROVIDER=groq`, `GROQ_API_KEY=...` (key: console.groq.com/keys) |
| **Ollama** | free, runs on your laptop | install ollama.com → `ollama pull qwen3:8b` → `LLM_PROVIDER=ollama` |
| OpenAI | paid | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...` |
| Anthropic | paid | `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY=...` |

All five speak the same "OpenAI-compatible" format, so `deepsearch/llm.py` uses one client for every provider.
Default model names were checked in **November 2026**. If you get a *model not found* error:
1. Open the provider's model list.
2. Set `LLM_MODEL=` to a current model name.

With a real LLM, the planner, query writer, claim extractor, judge and final answer are all done by the model.
Run the same labs again and compare the trace with mock mode.

## 5. Switch on real web search (optional)

| `SEARCH_PROVIDER` | Key? | Notes |
|---|---|---|
| `offline` (default) | no | Fictional corpus, repeatable results |
| `ddgs` | no | Real web results through the `ddgs` metasearch library |
| `tavily` | yes (free monthly credits) | Search API built for agents; set `TAVILY_API_KEY` |

Pages are fetched and cleaned with `trafilatura`.

For a real company, describe it precisely so the identity check works:
```bash
python run_agent.py "Is <Company> a reliable payment partner today?" \
  --name "<Company>" --city Mumbai --country India \
  --industry "payment gateway,payments" --domain company.com
```

> The report schema (`deepsearch/schemas.py → FieldName`) targets **payment / fintech vendor due diligence**.
> For another domain, edit three things: `FieldName`, `FIELD_DESCRIPTIONS`, and the regex rules in `extractor.py` (used in mock mode only).
>
> **Research companies, not private people.** See Module 11 of the course on India's DPDP Act and GDPR.

---

## 6. Project map

```
shivank-deep-search-agent/
├── run_agent.py              ★ capstone CLI
├── deepsearch/
│   ├── config.py             settings, provider presets (.env)
│   ├── schemas.py            Pydantic contracts: plan, evidence, report
│   ├── llm.py                one LLM wrapper: mock or live, validate→repair loop, tool calling
│   ├── tools.py              search_web / fetch_page / think — offline BM25 or live ddgs/tavily
│   ├── extractor.py          page → claims (LLM with spotlighting, or regex in mock mode)
│   ├── security.py           prompt-injection scan, hidden-text stripping, spotlighting
│   ├── planner.py            Plan Creator, Plan Reviewer, Plan Evaluator
│   ├── executor.py           parallel workers (orchestrator–worker pattern)
│   ├── validator.py          authority, freshness, identity, completeness, copy detection
│   ├── merger.py             dedupe, conflict resolution, provenance
│   ├── memory.py             long-term lessons + source track record
│   ├── budget.py             tokens / $ / time / tool-call limits
│   ├── evaluation.py         metrics + LLM-as-judge rubric
│   ├── baselines.py          naive search and basic RAG, for comparison
│   ├── tracing.py            trace.jsonl writer
│   └── orchestrator.py       the lead agent: plan → loop → synthesise → learn
├── labs/                     lab01 … lab12 (+ _common.py helpers)
├── data/corpus.json          the fictional web (19 pages)
├── data/gold_set.json        hand-checked answers for evaluation
└── tests/test_course.py      python -m pytest -q
```

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: deepsearch` | Run commands from the project root. Check the venv is activated. |
| `ModuleNotFoundError: mcp.server.fastmcp` | You are reading an old tutorial. MCP SDK v2 renamed FastMCP to `MCPServer` (`from mcp.server import MCPServer`). Our code already uses v2. To stay on v1: `pip install "mcp<2"` and change the import as shown in `lab05_mcp_server.py`. |
| `RuntimeError: asyncio.run() cannot be called from a running event loop` (Jupyter) | Run the labs as scripts in a terminal, or pass `parallel=False` to `DeepSearchAgent`. |
| `needs an API key` | Put the key in `.env` with the exact variable name from `.env.example`. |
| 404 / *model not found* | Set `LLM_MODEL` to a model your provider currently lists. |
| 429 / rate limit (free tiers) | Set `DS_MAX_PARALLEL=1` and try again, or wait a minute. |
| `400 ... temperature` | Remove `LLM_TEMPERATURE` from `.env`. Many reasoning models only accept their default. |
| Live search returns nothing | `ddgs` is sometimes rate-limited. Try again later, or switch to `tavily`. |
| Windows shows strange characters | Run `chcp 65001` once in the terminal, or use Windows Terminal. |
| Reset the agent's memory | Delete the `memory/` folder. |

---

© Shivank Agrawal · Future with Shivank. For learning. All companies, people and pages in `data/` are fictional.
