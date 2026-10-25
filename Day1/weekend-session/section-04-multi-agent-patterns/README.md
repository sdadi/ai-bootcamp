# Section 4 — Multi-Agent Patterns

| Notebook | Pattern | Example |
|---|---|---|
| `01_supervisor_worker.ipynb` | Supervisor–worker | Market research: news, financial and sentiment workers coordinated by a supervisor |
| `02_agent_as_tool.ipynb` | Agent as tool | Tutoring: a teacher delegates to math, writing and science tutors |
| `03_graph_pattern.ipynb` | Graph | E-commerce orders routed through risk, fraud, inventory, shipping and refund steps |
| `04_swarm_pattern.ipynb` | Swarm | Game level design: four specialists hand work to each other |

## Run

```bash
pip install -r requirements.txt
export TAVILY_API_KEY="tvly-..."          # notebook 1 (free key at tavily.com), or put it in the repo-root .env
export ALPHA_VANTAGE_API_KEY="..."        # optional
jupyter lab
```

## Run on Groq instead of Bedrock

Each notebook has a `_groq` twin; install `requirements-groq.txt` instead of `requirements.txt`. Set `TAVILY_API_KEY` in your shell or the repo-root `.env` for notebook 1.

- Notebook 1's supervisor prompt has one extra line: gpt-oss tends to ask the user to narrow a broad question instead of delegating, so it is told to pick sensible defaults and delegate straight away.
- Notebook 1 is the heaviest lab on Groq's free tier, which rejects any single request over 8,000 tokens. So its workers search with a compact `web_search` tool (Tavily, top 3 results, at most two searches per task) instead of the full `tavily_search`, and the "try different questions" cell asks one extra question with a fresh supervisor conversation. It takes about 15 minutes, and each question uses ~30–55k tokens: run it on a model with plenty of its 200k daily tokens left. `qwen/qwen3.8-27b` works (slower: 1,000 output tokens/minute); `openai/gpt-oss-20b` often sends malformed tool calls here.
- All four notebooks make many model calls, so on the free tier they run slower than on Bedrock, with automatic `rate limit` retries along the way. If you run out of daily tokens, switch `GROQ_MODEL_ID` — or run them on a local model with no limits at all (see "Run on a local model" in the repo README), at the cost of speed.

## Choosing a pattern

| Pattern | Who decides the route | Use when |
|---|---|---|
| Supervisor–worker | A central supervisor | Parallel research; combining specialist answers |
| Agent as tool | The orchestrator, which stays in control | Simple delegation |
| Graph | Your code (conditional edges) | Fixed, auditable procedures |
| Swarm | The agents, via handoffs | Creative or iterative work where the path can't be predicted |
