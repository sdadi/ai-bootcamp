"""
Send a Strands agent's traces to the local OpenTelemetry Collector.

The agent only knows about ONE endpoint (the collector on localhost:4318).
Where the traces end up — console, Datadog, Langfuse, all of them — is decided
by the collector config, not by your agent code.

Run:
    ./run_collector.sh debug          # or: ./run_collector.sh fanout
    python agent_via_collector.py
    docker logs otel-collector        # (debug mode) see the spans
"""

import os

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.telemetry import StrandsTelemetry
from strands_tools import calculator

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
os.environ.setdefault("OTEL_SERVICE_NAME", "strands-agent-via-collector")

telemetry = StrandsTelemetry().setup_otlp_exporter()

agent = Agent(
    model=BedrockModel(model_id=MODEL_ID, region_name=REGION),
    system_prompt="You are a helpful finance assistant.",
    tools=[calculator],
)
agent("A laptop costs Rs 84,999 before 18% GST. What is the final price and the GST amount?")

telemetry.tracer_provider.force_flush()
print("\n\nTraces sent to the collector at", os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"])
