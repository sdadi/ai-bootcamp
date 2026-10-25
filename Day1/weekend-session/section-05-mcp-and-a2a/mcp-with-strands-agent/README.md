# Strands Agents + MCP Servers

| Script | MCP server it uses | Try asking |
|---|---|---|
| `agent_with_local_mcp.py` | `../mcp-basics/server.py` (stdio) | Weather in Bangalore and London |
| `agent_with_aws_docs_mcp.py` | [AWS Documentation MCP server](https://github.com/awslabs/mcp) via `uvx` | "What is AWS Lambda and what is its max timeout?" |

The pattern is the same in both:

```python
mcp_client = MCPClient(lambda: stdio_client(StdioServerParameters(command=..., args=[...])))

with mcp_client:
    tools = mcp_client.list_tools_sync()
    agent = Agent(tools=tools)
    agent("...")
```

## Run

```bash
pip install -r requirements.txt     # includes uv, which provides uvx
python agent_with_local_mcp.py
python agent_with_aws_docs_mcp.py   # first run downloads the docs server (~30 s)
```

## Run on Groq instead of Bedrock

```bash
pip install -r requirements-groq.txt
python agent_with_local_mcp_groq.py
python agent_with_aws_docs_mcp_groq.py
```

Same agents and MCP servers; the model runs on Groq (`GROQ_API_KEY`, `GROQ_MODEL_ID`). No AWS credentials are needed for either script.
