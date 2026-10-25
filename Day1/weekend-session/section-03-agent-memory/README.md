# Section 3 — Agent Memory

| Notebook | What it covers |
|---|---|
| `01_personalized_memory_mem0.ipynb` | Long-term, per-user memory with Mem0. Each agent's memory tool is bound to one user in code, so users are isolated even against prompt injection. Plus an admin "super-agent" with an explicit cross-user tool. |
| `02-langgraph-agentcore-memory/math_agent_with_checkpointing.ipynb` | Short-term (session) memory for a LangGraph agent, using Amazon Bedrock AgentCore Memory as the checkpointer. |

## Run

```bash
# Notebook 1
pip install -r requirements.txt

# Notebook 2
pip install -r 02-langgraph-agentcore-memory/requirements.txt
```

Notebook 1 stores memories locally (FAISS) and uses Bedrock for embeddings (Titan v2) and fact extraction (Claude Haiku 4.5). Notebook 2 creates an AgentCore Memory resource and deletes it at the end.

## Run on Groq instead of Bedrock

| Bedrock | Groq | Needs |
|---|---|---|
| `01_personalized_memory_mem0.ipynb` | `01_personalized_memory_mem0_groq.ipynb` | only a Groq key — Mem0 extracts facts with `openai/gpt-oss-20b` and embeds them **locally** with fastembed (`mixedbread-ai/mxbai-embed-large-v1`, ~640 MB download on first run), because Groq has no embeddings API |
| `02-.../math_agent_with_checkpointing.ipynb` | `02-.../math_agent_with_checkpointing_groq.ipynb` | AWS credentials for AgentCore Memory (no Bedrock model access) |

```bash
pip install -r requirements-groq.txt                                  # notebook 1
pip install -r 02-langgraph-agentcore-memory/requirements-groq.txt    # notebook 2
```

**IAM for notebook 2:** `bedrock-agentcore:CreateMemory`, `GetMemory`, `ListMemories`, `DeleteMemory`, `CreateEvent`, `ListEvents`, `GetEvent`, `DeleteEvent`.
