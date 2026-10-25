"""Document analysis agent for Amazon Bedrock AgentCore Runtime.

Compared with a plain Strands agent, only four things are added:
  1. import BedrockAgentCoreApp
  2. app = BedrockAgentCoreApp()
  3. @app.entrypoint on the handler
  4. app.run() in __main__

Local test:
    python agent.py
    curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" \
         -d '{"prompt": "What are the key clauses in a standard NDA?"}'
"""

import os

from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

app = BedrockAgentCoreApp()

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")


@tool
def extract_clauses(text: str) -> dict:
    """Extract key clauses from contract text."""
    # In production this would use NLP or a specialised model.
    # For this example we return a structured placeholder.
    return {"clauses": ["termination", "liability", "indemnification"]}


model = BedrockModel(model_id=MODEL_ID)
agent = Agent(
    model=model,
    tools=[extract_clauses],
    system_prompt="You are a document analysis assistant. Extract and summarize key contract clauses.",
)


@app.entrypoint
def handle(payload):
    response = agent(payload.get("prompt"))
    return response.message["content"][0]["text"]


if __name__ == "__main__":
    app.run()
