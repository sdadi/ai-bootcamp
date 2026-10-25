import json
import os

from strands import Agent
from strands.models.openai import OpenAIModel

# Groq edition of app.py: only the model provider changes. GROQ_API_KEY comes from the
# function's environment (template-groq.yaml resolves it from Secrets Manager at deploy time).
MODEL_ID = os.environ.get("MODEL_ID") or os.environ.get("GROQ_MODEL_ID") or "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a document analysis assistant.
When given a contract or document, you:
1. Extract key terms and parties involved
2. Flag any risky or unusual clauses
3. Provide a clear summary

Be concise and specific. Cite the exact language from the document when flagging risks."""

# Initialize the model outside the handler so it is reused across warm invocations.
# The first request pays the setup cost; later requests on the same container reuse it.
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


def lambda_handler(event, context):
    """Handle API Gateway POST requests."""
    # Parse the request body
    if isinstance(event.get("body"), str):
        body = json.loads(event["body"])
    else:
        body = event.get("body") or event

    document_text = body.get("document_text", "")

    if not document_text:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Missing 'document_text' in request body"}),
        }

    # A fresh Agent per request: a warm container serves many different callers,
    # so conversation history must never carry over from one request to the next.
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, callback_handler=None)
    result = agent(f"Analyze this contract:\n\n{document_text}")

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"analysis": str(result)}),
    }
