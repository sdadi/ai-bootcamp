"""Section 1: Hello World agent with the Strands Agents SDK on Amazon Bedrock.

Run:
    python hello_agent.py
"""

import os

from strands import Agent
from strands.models.bedrock import BedrockModel

# One place to choose the model. Override without editing code:
#   export BEDROCK_MODEL_ID=global.anthropic.claude-haiku-4-5-20251001-v1:0
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

model = BedrockModel(model_id=MODEL_ID, region_name=REGION)
agent = Agent(model=model)

response = agent("Hello! Tell me a fun fact about AI agents.")
print()
print("-" * 60)
print(response)
