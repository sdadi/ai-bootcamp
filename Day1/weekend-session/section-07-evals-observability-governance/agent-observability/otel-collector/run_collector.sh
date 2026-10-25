#!/usr/bin/env bash
# Start the OpenTelemetry Collector (contrib build) in Docker.
#
#   ./run_collector.sh debug      # print spans to the collector console (no accounts needed)
#   ./run_collector.sh fanout     # send spans to Datadog AND Langfuse
#
# For "fanout", export these first:
#   export DD_API_KEY=...            DD_SITE=datadoghq.com   (or us5.datadoghq.com, datadoghq.eu, ...)
#   export LANGFUSE_PUBLIC_KEY=pk-lf-...   LANGFUSE_SECRET_KEY=sk-lf-...
#   export LANGFUSE_BASE_URL=https://cloud.langfuse.com   (or https://us.cloud.langfuse.com / your self-hosted URL)
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-debug}"
IMAGE="otel/opentelemetry-collector-contrib:latest"
docker rm -f otel-collector >/dev/null 2>&1 || true

if [ "$MODE" = "fanout" ]; then
  : "${DD_API_KEY:?Set DD_API_KEY}" "${LANGFUSE_PUBLIC_KEY:?Set LANGFUSE_PUBLIC_KEY}" "${LANGFUSE_SECRET_KEY:?Set LANGFUSE_SECRET_KEY}"
  LANGFUSE_AUTH=$(printf '%s:%s' "$LANGFUSE_PUBLIC_KEY" "$LANGFUSE_SECRET_KEY" | base64 | tr -d '\n')
  docker run -d --name otel-collector -p 4317:4317 -p 4318:4318 \
    -e DD_API_KEY -e DD_SITE="${DD_SITE:-datadoghq.com}" \
    -e LANGFUSE_OTEL_ENDPOINT="${LANGFUSE_BASE_URL:-https://cloud.langfuse.com}/api/public/otel" \
    -e LANGFUSE_AUTH="$LANGFUSE_AUTH" \
    -v "$PWD/otel-collector-config.yaml:/etc/otelcol-contrib/config.yaml:ro" \
    "$IMAGE"
else
  docker run -d --name otel-collector -p 4317:4317 -p 4318:4318 \
    -v "$PWD/otel-collector-config.debug.yaml:/etc/otelcol-contrib/config.yaml:ro" \
    "$IMAGE"
fi

echo "Collector running ($MODE). Logs: docker logs -f otel-collector   Stop: docker rm -f otel-collector"
