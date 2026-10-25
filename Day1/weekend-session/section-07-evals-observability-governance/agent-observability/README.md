# Agent Observability

Strands emits OpenTelemetry spans for every agent invocation, reasoning cycle, model call and tool call. You only choose where to send them.

| Notebook | Backend | Setup |
|---|---|---|
| `01_console_tracing.ipynb` | stdout | none |
| `02_jaeger_tracing.ipynb` | Jaeger on localhost | Docker |
| `03_langfuse_tracing.ipynb` | Langfuse (cloud or self-hosted) | Langfuse keys |
| `04_langsmith_tracing.ipynb` | LangSmith | LangSmith API key |
| `05_cloudwatch_adot_tracing.ipynb` | CloudWatch GenAI Observability via ADOT | Transaction Search enabled |
| `otel-collector/` | One collector fanning out to Datadog **and** Langfuse | Docker; Datadog + Langfuse keys (or the debug config) |

Run each notebook in a fresh kernel — OpenTelemetry allows one global tracer provider per process.

**No Bedrock access?** Each notebook has a `_groq` twin, plus `agentcore_agent_groq.py` and `otel-collector/agent_via_collector_groq.py`: `pip install -r requirements-groq.txt`. The spans are the same (`invoke_agent`, `execute_event_loop_cycle`, `chat`, `execute_tool ...`), with `gen_ai.request.model` showing the Groq model. Notebook 5 still needs AWS credentials for CloudWatch.

```bash
pip install -r requirements.txt
jupyter lab
```

## OpenTelemetry Collector fan-out (`otel-collector/`)

Point every agent at one collector; let the collector decide where traces go.

```
Strands agents ──OTLP──► OTel Collector ──┬──► Datadog   (alerting / APM)
                                          └──► Langfuse  (LLM trace analysis)
```

```bash
cd otel-collector
./run_collector.sh debug              # no accounts: spans printed by the collector
# or
export DD_API_KEY=... DD_SITE=datadoghq.com
export LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-... LANGFUSE_BASE_URL=https://cloud.langfuse.com
./run_collector.sh fanout

python agent_via_collector.py
docker logs otel-collector            # debug mode: see the spans
docker rm -f otel-collector           # stop
```

`otel-collector-config.yaml` is the fan-out configuration; `otel-collector-config.debug.yaml` is the no-account version.
