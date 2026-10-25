# Section 5 — MCP and A2A: How Agents Talk

| Folder | What it shows |
|---|---|
| `mcp-basics/` | An MCP server exposing every primitive — tools, resources, prompts, roots — and a client that uses them. No LLM involved. |
| `mcp-with-strands-agent/` | Strands agents that get their tools from MCP servers: our local server, and the AWS Documentation MCP server |
| `a2a-travel-agents/` | Two A2A agent servers (weather, flights) and an orchestrator that discovers and calls them |

## MCP vs A2A

| | MCP | A2A |
|---|---|---|
| Purpose | Connect an LLM app to tools and data | Connect agents to other agents |
| Transport | stdio, streamable HTTP | HTTP (JSON-RPC, SSE) |
| Discovery | `list_tools()`, `list_resources()`, `list_prompts()` | Agent card at `/.well-known/agent-card.json` |
| Who has the LLM | The client side | Both sides — each agent has its own |
| Messages | Structured calls (tool name + arguments) | Natural-language messages and tasks |

In short: **MCP is LLM ↔ tool. A2A is agent ↔ agent.**

## Run

```bash
# MCP basics (no AWS needed)
cd mcp-basics && pip install -r requirements.txt && python client.py

# Strands + MCP
cd ../mcp-with-strands-agent && pip install -r requirements.txt
python agent_with_local_mcp.py
python agent_with_aws_docs_mcp.py

# A2A — three terminals
cd ../a2a-travel-agents && pip install -r requirements.txt
python weather_agent_server.py      # terminal 1
python flights_agent_server.py      # terminal 2
python travel_orchestrator.py       # terminal 3
```

## Run on Groq instead of Bedrock

Every script that calls a model has a `_groq` twin; `mcp-basics/` needs no model at all, and the A2A orchestrator only calls the agents, so it is unchanged.

```bash
cd mcp-with-strands-agent && pip install -r requirements-groq.txt
python agent_with_local_mcp_groq.py
python agent_with_aws_docs_mcp_groq.py      # no AWS credentials needed: the docs are public

cd ../a2a-travel-agents && pip install -r requirements-groq.txt
python weather_agent_server_groq.py         # terminal 1
python flights_agent_server_groq.py         # terminal 2
python travel_orchestrator.py               # terminal 3 (same orchestrator)
```
