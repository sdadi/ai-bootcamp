# Agentic RAG with DeepSeek + CrewAI

*Created by Shivank Agrawal*

A team of 5 AI agents that answers questions from a private PDF, goes to the
internet when the PDF doesn't have the answer, and checks its own work for
relevance and hallucination before replying.

Read `../Agentic_RAG_Guide.html` first. It explains every idea and every line here.

## Files, in the order to run them

| File | What it does |
|---|---|
| `step0_simulate_flow.py` | Pure Python, no keys, no installs. Shows the whole agent flow with toy logic. |
| `step1_plain_rag.py` | Traditional RAG, no agents: search the PDF, one LLM call. The "before" picture. |
| `main.py` | The full agentic RAG crew. Single question or `--chat` mode. |
| `config.py` | Reads keys and settings from `.env`. One place for every setting. |
| `llms.py` | Creates the two LLMs: the main brain and the fast RAG LLM. |
| `tools.py` | The two tools: `rag_tool` (search the PDF) and `web_search_tool` (Tavily). |
| `agents.py` | The 5 agents: Router, Retriever, Answer Grader, Hallucination Grader, Answer Generator. |
| `tasks.py` | The 5 tasks and how information flows between them (`context`). |

## Setup (10–15 minutes)

```bash
cd code
python -m venv venv
# Windows (PowerShell):   venv\Scripts\Activate.ps1
# Windows (Git Bash):     source venv/Scripts/activate
# Mac / Linux:            source venv/bin/activate

pip install -r requirements.txt     # first time takes a few minutes (installs PyTorch)
```

Copy `.env.example` to `.env` and paste your keys.

| Key | Where | Needed for |
|---|---|---|
| `OPENROUTER_API_KEY` | openrouter.ai → Keys. Add a small credit (DeepSeek costs a tiny fraction of a rupee per question). | `LLM_MODE=openrouter` (default) |
| `GROQ_API_KEY` | console.groq.com → API Keys (free tier) | the RAG LLM, and `LLM_MODE=groq` |
| `TAVILY_API_KEY` | app.tavily.com (free tier) | web search |

**No payment at all?** Set `LLM_MODE=groq` (fast free model, not DeepSeek) or
`LLM_MODE=ollama` (DeepSeek on your laptop).

## Run

```bash
python step0_simulate_flow.py                              # feel the flow (no keys)
python step1_plain_rag.py "What is multi-head attention?"   # plain RAG
python main.py                                              # agentic RAG, default question
python main.py "What is the latest version of Python?"      # goes to the web
python main.py --chat                                       # ask many questions
```

Each answer ends with a one-line journey, for example
`route=vectorstore  relevant=yes  grounded=yes  time=41s`, and is saved to `answers_log.md`.

The first run downloads the embedding model (~130 MB) and the PDF and builds the
vector database. Later runs are faster.

## DeepSeek on your laptop (Ollama)

1. Install Ollama from ollama.com
2. `ollama pull deepseek-r1:8b` (about 5 GB, needs ~8 GB free RAM)
3. In `.env`: `LLM_MODE=ollama` and optionally `RAG_LLM_MODE=same`
4. `python main.py`

Smaller models such as `deepseek-r1:1.5b` often fail at tool calls. Use 8b or bigger.

## Common errors

| Error | Fix |
|---|---|
| `Please fix your .env file first` | A key or setting is missing. Check `.env` (not `.env.example`). |
| `404` / model not found | Model IDs change. Pick a current one on the provider's models page and update `.env`. |
| `402` / insufficient credits (OpenRouter) | Add a small credit, or switch to `LLM_MODE=groq`. |
| `429` rate limit | Free tiers allow few requests per minute. Wait, or lower `MAX_RPM`. |
| `does not support tools` (Ollama) | Use `deepseek-r1:8b` or bigger. |
| PDF questions go to the web | Improve `PDF_TOPIC` so the Router knows what the PDF contains. |

## Use your own PDF

1. Put it in `data/`, e.g. `data/leave_policy.pdf`
2. In `.env`: `PDF_PATH=data/leave_policy.pdf`
3. In `.env`: `PDF_TOPIC=our company's leave, travel and reimbursement policy`
4. `python main.py "How many casual leaves do I get in a year?"`
