"""Section 1: Hello World agent with LangGraph / LangChain on Amazon Bedrock.

`create_agent` (from langchain.agents) is the current way to build a
ReAct-style agent that runs on the LangGraph runtime. It replaces the older
`langgraph.prebuilt.create_react_agent`.

Run:
    python hello_langgraph_agent.py
"""

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))


# A simple tool the agent can call
@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Welcome to the world of AI agents."


# Initialize the LLM via the Bedrock Converse API
llm = init_chat_model(MODEL_ID, model_provider="bedrock_converse", region_name=REGION)

# Create a ReAct agent with the tool
agent = create_agent(model=llm, tools=[greet], system_prompt="You are a friendly assistant.")

# Run the agent
response = agent.invoke({"messages": [{"role": "user", "content": "Please greet Alice and Bob."}]})

# Print every message in the conversation: user -> AI (tool calls) -> tool results -> AI answer
for message in response["messages"]:
    print(f"{message.type}: {message.text}")
