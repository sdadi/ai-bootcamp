import json
import os

from strands import Agent
from strands.models.bedrock import BedrockModel

MODEL_ID = os.environ.get("MODEL_ID") or os.environ.get("BEDROCK_MODEL_ID") or "global.anthropic.claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a document analysis assistant.
When given a contract or document, you:
1. Extract key terms and parties involved
2. Flag any risky or unusual clauses
3. Provide a clear summary

Be concise and specific. Cite the exact language from the document when flagging risks."""

# Initialize the model outside the handler so it is reused across warm invocations.
# The first request pays the setup cost; later requests on the same container reuse it.
model = BedrockModel(model_id=MODEL_ID)


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
