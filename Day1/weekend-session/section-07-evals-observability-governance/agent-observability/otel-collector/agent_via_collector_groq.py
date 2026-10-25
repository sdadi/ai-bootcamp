"""
Send a Strands agent's traces to the local OpenTelemetry Collector.

The agent only knows about ONE endpoint (the collector on localhost:4318).
Where the traces end up — console, Datadog, Langfuse, all of them — is decided
by the collector config, not by your agent code.

Run:
    ./run_collector.sh debug          # or: ./run_collector.sh fanout
    python agent_via_collector_groq.py
    docker logs otel-collector        # (debug mode) see the spans

Groq edition of agent_via_collector.py: only the model provider changes (needs GROQ_API_KEY).
"""

import os

from dotenv import load_dotenv
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.telemetry import StrandsTelemetry
from strands_tools import calculator

load_dotenv()

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")

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

os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
os.environ.setdefault("OTEL_SERVICE_NAME", "strands-agent-via-collector")

telemetry = StrandsTelemetry().setup_otlp_exporter()

agent = Agent(
    model=model,
    system_prompt="You are a helpful finance assistant.",
    tools=[calculator],
)
agent("A laptop costs Rs 84,999 before 18% GST. What is the final price and the GST amount?")

telemetry.tracer_provider.force_flush()
print("\n\nTraces sent to the collector at", os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])
