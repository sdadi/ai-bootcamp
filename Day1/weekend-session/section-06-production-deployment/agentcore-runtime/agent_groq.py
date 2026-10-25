"""Document analysis agent for Amazon Bedrock AgentCore Runtime, powered by Groq.

Same agent as agent.py; only the model provider changes. Groq exposes an
OpenAI-compatible API, so Strands' OpenAIModel talks to it via `base_url`.
The deployed runtime needs GROQ_API_KEY in its environment (see the notebook).

Compared with a plain Strands agent, only four things are added:
  1. import BedrockAgentCoreApp
  2. app = BedrockAgentCoreApp()
  3. @app.entrypoint on the handler
  4. app.run() in __main__

Local test:
    python agent_groq.py      (needs GROQ_API_KEY, or a .env file)
    curl -X POST http://localhost:8080/invocations -H "Content-Type: application/json" \
         -d '{"prompt": "What are the key clauses in a standard NDA?"}'
"""

import os

from bedrock_agentcore import BedrockAgentCoreApp
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.openai import OpenAIModel

load_dotenv()

app = BedrockAgentCoreApp()

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")


@tool
def extract_clauses(text: str) -> dict:
    """Extract key clauses from contract text."""
    # In production this would use NLP or a specialised model.
    # For this example we return a structured placeholder.
    return {"clauses": ["termination", "liability", "indemnification"]}


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
