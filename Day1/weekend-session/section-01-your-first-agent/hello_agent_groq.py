"""Section 1: Hello World agent with the Strands Agents SDK on Groq.

Same agent as hello_agent.py; only the model provider changes. Groq exposes an
OpenAI-compatible API, so Strands' OpenAIModel talks to it via `base_url`.

Run:
    export GROQ_API_KEY=...   # or put it in a .env file
    python hello_agent_groq.py
"""

import os

from dotenv import load_dotenv
from strands import Agent
from strands.models.openai import OpenAIModel

load_dotenv()

# --- FORCE ENVIRONMENT CLEAN UP ---
# This wipes out any hidden, cached environment variables causing the 404
os.environ.pop("GROQ_BASE_URL", None)
os.environ.pop("GROQ_LLM_MODEL", None)
os.environ.pop("GROQ_MODEL_ID", None)

# One place to choose the model. Override without editing code:
#   export GROQ_MODEL_ID=openai/gpt-oss-20b
MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")

model = OpenAIModel(
    client_args={
        "api_key": os.environ["GROQ_API_KEY"],
        "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),  # or a local OpenAI-compatible server
        "max_retries": 5,  # retry dropped connections (the client default is 2)
    },
    model_id=MODEL_ID,
    # Groq reasoning models stream their thinking by default; hide it so only the answer prints.
    params={"extra_body": {"include_reasoning": False}},
)
agent = Agent(model=model)

response = agent("Hello! Tell me a fun fact about AI agents.")
print()
print("-" * 60)
print(response)
