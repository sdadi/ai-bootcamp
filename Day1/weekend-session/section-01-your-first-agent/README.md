# Section 1 — Your First AI Agent

Build your first agents on Amazon Bedrock with two frameworks.

| File | What it does |
|---|---|
| `01_hello_agent.ipynb` | **Start here.** Model choice → one-line agent → your own tool → a prebuilt tool → system prompt |
| `hello_agent.py` | The smallest possible Strands agent as a script |
| `hello_langgraph_agent.py` | A ReAct agent with one tool, built with LangChain/LangGraph (`create_agent`) |

## Run

```bash
pip install -r requirements.txt
jupyter lab 01_hello_agent.ipynb

python hello_agent.py
python hello_langgraph_agent.py
```

## Run on Groq instead of Bedrock

No AWS account? Every file has a `_groq` twin that runs the same agent code on [Groq](https://console.groq.com). Strands' `OpenAIModel` points at Groq's OpenAI-compatible endpoint, and LangChain uses `langchain-groq`.

| Bedrock | Groq |
|---|---|
| `01_hello_agent.ipynb` | `01_hello_agent_groq.ipynb` |
| `hello_agent.py` | `hello_agent_groq.py` |
| `hello_langgraph_agent.py` | `hello_langgraph_agent_groq.py` |
| `requirements.txt` | `requirements-groq.txt` |

```bash
pip install -r requirements-groq.txt
export GROQ_API_KEY=...        # or put GROQ_API_KEY=... in a .env file
jupyter lab 01_hello_agent_groq.ipynb

python hello_agent_groq.py
python hello_langgraph_agent_groq.py
```

The default model is `openai/gpt-oss-120b`; set `GROQ_MODEL_ID` to use another one (for example, `openai/gpt-oss-20b`).

## Strands vs LangChain vs LangGraph

| | Strands Agents | LangChain | LangGraph |
|---|---|---|---|
| Approach | Agent-first SDK: model + tools + prompt | Building blocks: prompts, parsers, loaders, integrations | Stateful graph orchestration |
| Boilerplate | Minimal | Moderate | More setup, more control |
| State | Built in (+ AgentCore Memory / session managers) | Manual | Built in with checkpointers |
| Best for | Fast, AWS-native agents | RAG pipelines and integrations | Complex flows with loops, branches, human-in-the-loop |

## When to use which

```
Need an agent fast on AWS?                    → Strands
Need a simple LLM chain (no agent loop)?      → LangChain
Need complex orchestration or multi-agent?    → LangGraph (or Strands graph/swarm — Section 4)
Need persistent state across sessions?        → LangGraph or Strands, with AgentCore Memory (Section 3)
Going to production on AWS?                   → Either, deployed on AgentCore Runtime (Section 6)
```
