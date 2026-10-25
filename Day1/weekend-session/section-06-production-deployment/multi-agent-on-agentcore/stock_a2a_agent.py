"""
Stock A2A Agent — answers finance questions with MCP stock tools + web requests.

Combines the MCP stock price tool (from stock_mcp_server.py) with the Strands
http_request tool, and serves the agent over the A2A protocol.

Local:
    1. python stock_mcp_server.py
    2. python stock_a2a_agent.py            (http://127.0.0.1:9001)

Deploy: agentcore configure -e stock_a2a_agent.py --protocol A2A ...
        (on AgentCore, A2A agents must listen on port 9000)
"""

import logging
import os

import uvicorn
from a2a.types import AgentSkill
from fastapi import FastAPI
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.multiagent.a2a import A2AServer
from strands.tools.mcp import MCPClient
from strands_tools import http_request

logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
STOCK_MCP_URL = os.environ.get("STOCK_MCP_URL", "http://localhost:8000/mcp")
# Set automatically on AgentCore Runtime; used as the public URL in the agent card
RUNTIME_URL = os.environ.get("AGENTCORE_RUNTIME_URL", "http://127.0.0.1:9001/")
# Bearer token for calling the MCP server when it is deployed behind OAuth
BEARER_TOKEN = os.environ.get("AGENT_BEARER_TOKEN")

# Locally this agent uses 9001 (the orchestrator has 9000). On AgentCore Runtime,
# A2A agents must listen on 9000 — the deploy command passes PORT=9000.
PORT = int(os.environ.get("PORT", "9000" if os.environ.get("DOCKER_CONTAINER") else "9001"))

MCP_HEADERS = {"Authorization": f"Bearer {BEARER_TOKEN}"} if BEARER_TOKEN else None

# MCP client for the stock price tools. Passing it straight into tools=[...]
# lets the agent manage the connection for us.
mcp_client = MCPClient(lambda: streamablehttp_client(STOCK_MCP_URL, headers=MCP_HEADERS))

stock_agent = Agent(
    name="Stock Agent",
    description="Provides stock prices and financial information.",
    model=BedrockModel(model_id=MODEL_ID),
    system_prompt=(
        "You are a financial assistant. Use get_stock_price to look up current stock prices. "
        "Use http_request to fetch data from financial websites when needed. "
        "Always provide the ticker symbol, current price, and any relevant context."
    ),
    tools=[mcp_client, http_request],
    callback_handler=None,
)

SKILLS = [
    AgentSkill(
        id="stock_lookup",
        name="Stock Price Lookup",
        description="Get latest stock prices and financial info for any publicly traded company.",
        tags=["stocks", "finance", "prices", "market"],
        examples=[
            "What is the current price of AAPL?",
            "How is AMZN doing today?",
            "Get me the stock price for MSFT and GOOGL",
        ],
    ),
]

a2a_server = A2AServer(
    agent=stock_agent,
    http_url=RUNTIME_URL,
    serve_at_root=True,
    skills=SKILLS,
    version="1.0.0",
    enable_a2a_compliant_streaming=True,
)

app = FastAPI()


@app.get("/ping")
def ping():
    return {"status": "healthy"}


app.mount("/", a2a_server.to_fastapi_app())

if __name__ == "__main__":
    print(f"Stock A2A Agent on http://0.0.0.0:{PORT} (MCP: {STOCK_MCP_URL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
