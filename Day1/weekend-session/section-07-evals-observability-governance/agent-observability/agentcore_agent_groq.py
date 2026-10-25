"""Customer account agent used by the CloudWatch / ADOT tracing notebook.

The notebook runs it with `opentelemetry-instrument python agentcore_agent_groq.py`
so the AWS Distro for OpenTelemetry can capture traces at the process level.

Groq edition of agentcore_agent.py: only the model provider changes (needs GROQ_API_KEY).
"""

import os

from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.openai import OpenAIModel

load_dotenv()

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")


@tool
def lookup_account(account_id: str) -> str:
    """Look up account details by account ID."""
    # Simulated account data
    accounts = {
        "ACC-001": {"name": "Acme Corp", "tier": "Enterprise", "balance": 142500.00},
        "ACC-002": {"name": "Startup Inc", "tier": "Growth", "balance": 28750.00},
    }
    account = accounts.get(account_id)
    if account:
        return f"Account {account_id}: {account['name']}, Tier: {account['tier']}, Balance: ${account['balance']:,.2f}"
    return f"Account {account_id} not found."


@tool
def calculate_discount(amount: float, tier: str) -> str:
    """Calculate discount based on customer tier."""
    rates = {"Enterprise": 0.15, "Growth": 0.10, "Starter": 0.05}
    rate = rates.get(tier, 0.0)
    discount = amount * rate
    return f"{tier} tier discount: {rate * 100:.0f}% off ${amount:,.2f} = ${discount:,.2f} savings"


model = OpenAIModel(
    client_args={
        "api_key": os.environ["GROQ_API_KEY"],
        "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),  # or a local OpenAI-compatible server
        "max_retries": 5,  # retry dropped connections (the client default is 2)
    },
    model_id=MODEL_ID,
    # Groq reasoning models stream their thinking by default; hide it so only the answer is returned.
    params={"extra_body": {"include_reasoning": False}},
)

agent = Agent(
    model=model,
    system_prompt="You are a customer account assistant. Use the available tools to help with account inquiries.",
    tools=[lookup_account, calculate_discount],
)

response = agent("Look up account ACC-001 and calculate what discount they would get on a $50,000 purchase.")
print(response)
