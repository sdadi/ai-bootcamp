"""
Finance Orchestrator — an A2A agent that delegates stock questions to the
Stock Agent over A2A.

Local:
    1. python stock_mcp_server.py
    2. python stock_a2a_agent_groq.py
    3. python orchestrator_groq.py               (http://127.0.0.1:9000)

Deploy: agentcore configure -e orchestrator_groq.py --protocol A2A ...

Groq edition of orchestrator.py: only the model provider changes (needs GROQ_API_KEY, or a .env file locally).
"""

import logging
import os

import httpx
import uvicorn
from a2a.client import ClientConfig
from a2a.types import AgentSkill
from fastapi import FastAPI
from dotenv import load_dotenv
from strands import Agent, tool
from strands.agent.a2a_agent import A2AAgent
from strands.models.openai import OpenAIModel
from strands.multiagent.a2a import A2AServer

logging.basicConfig(level=logging.INFO)

load_dotenv()

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")

model = OpenAIModel(
    client_args={
        "api_key": os.environ["GROQ_API_KEY"],
        "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        "max_retries": 5,  # retry dropped connections (the client default is 2)
    },
    model_id=MODEL_ID,
    # Groq reasoning models stream their thinking by default; hide it so only the answer is returned.
    params={"extra_body": {"include_reasoning": False}},
)
RUNTIME_URL = os.environ.get("AGENTCORE_RUNTIME_URL", "http://127.0.0.1:9000/")
STOCK_A2A_URL = os.environ.get("STOCK_A2A_URL", "http://127.0.0.1:9001")
# Bearer token for calling the Stock Agent when it is deployed behind OAuth
BEARER_TOKEN = os.environ.get("AGENT_BEARER_TOKEN")
PORT = int(os.environ.get("PORT", "9000"))


def remote_stock_agent() -> A2AAgent:
    """A client for the remote Stock Agent (adds the bearer token when deployed)."""
    if BEARER_TOKEN:
        http = httpx.AsyncClient(headers={"Authorization": f"Bearer {BEARER_TOKEN}"}, timeout=300)
        return A2AAgent(STOCK_A2A_URL, client_config=ClientConfig(httpx_client=http))
    return A2AAgent(STOCK_A2A_URL, timeout=300)


@tool
def ask_stock_agent(question: str) -> str:
    """Ask the Stock Agent (a remote A2A agent) for current stock prices and market information.

    Args:
        question: A natural-language question, e.g. "What is the current price of NVDA?"
    """
    return str(remote_stock_agent()(question))


orchestrator = Agent(
    name="Finance Orchestrator",
    description="Provides financial advice by coordinating with the Stock agent.",
    model=model,
    system_prompt=(
        "You are a financial advisor assistant. When users ask about stocks, "
        "investments, or market conditions, use ask_stock_agent to get current prices. "
        "Provide helpful analysis and context with the data. This is not financial advice."
    ),
    tools=[ask_stock_agent],
    callback_handler=None,
)

SKILLS = [
    AgentSkill(
        id="finance_advisor",
        name="Finance Advisor",
        description="Answers finance questions using real-time stock data.",
        tags=["finance", "stocks", "investing", "market"],
        examples=[
            "How are the FAANG stocks doing?",
            "Should I look at NVDA right now?",
            "Compare AAPL and MSFT prices",
        ],
    ),
]

a2a_server = A2AServer(
    agent=orchestrator,
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
    print(f"Finance Orchestrator on http://0.0.0.0:{PORT} (Stock Agent: {STOCK_A2A_URL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
