"""Section 1: Hello World agent with LangGraph / LangChain on Groq.

Same agent as hello_langgraph_agent.py; only the model provider changes
(`bedrock_converse` -> `groq`, which needs the langchain-groq package).

Run:
    export GROQ_API_KEY=...   # or put it in a .env file
    python hello_langgraph_agent_groq.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

# --- FORCE ENVIRONMENT CLEAN UP ---
# This wipes out any hidden, cached environment variables causing the 404
os.environ.pop("GROQ_BASE_URL", None)
os.environ.pop("GROQ_LLM_MODEL", None)
os.environ.pop("GROQ_MODEL_ID", None)

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")


# A simple tool the agent can call
@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Welcome to the world of AI agents."


# Initialize the LLM via Groq (reads GROQ_API_KEY from the environment)
llm = init_chat_model(MODEL_ID, model_provider="groq", max_retries=5)

# Create a ReAct agent with the tool
agent = create_agent(model=llm, tools=[greet], system_prompt="You are a friendly assistant.")

# Run the agent
response = agent.invoke({"messages": [{"role": "user", "content": "Please greet Alice and Bob."}]})

# Print every message in the conversation: user -> AI (tool calls) -> tool results -> AI answer
for message in response["messages"]:
    print(f"{message.type}: {message.text}")
