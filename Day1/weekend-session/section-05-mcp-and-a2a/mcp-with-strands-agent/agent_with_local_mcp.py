"""
A Strands agent that gets its tools from an MCP server.

It starts the server from ../mcp-basics/server.py over stdio, discovers its
tools, and lets the model decide when to call them. The agent code knows
nothing about the weather tool in advance — it all comes over MCP.

Run:  python agent_with_local_mcp.py
"""

import os
import sys
from pathlib import Path

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

SERVER = Path(__file__).resolve().parent.parent / "mcp-basics" / "server.py"

# MCPClient takes a function that opens the transport. Here: spawn the server.
mcp_client = MCPClient(
    lambda: stdio_client(StdioServerParameters(command=sys.executable, args=[str(SERVER)]))
)

with mcp_client:
    tools = mcp_client.list_tools_sync()
    print("Tools discovered over MCP:", [t.tool_name for t in tools])

    agent = Agent(model=BedrockModel(model_id=MODEL_ID, region_name=REGION), tools=tools)
    agent("I'm flying from Bangalore to London tomorrow. What's the weather like in both cities?")
    print()
