# Reference copy of the Dockerfile the AgentCore starter toolkit generates.
# `runtime.configure(...)` / `agentcore configure` regenerates this file, so
# you normally never edit it by hand. It is kept here so you can see what
# gets built: an ARM64 Python image, your requirements, the AWS OpenTelemetry
# distro for observability, a non-root user, and your agent as the entrypoint.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1 \
    AWS_REGION=us-east-1 \
    AWS_DEFAULT_REGION=us-east-1

COPY requirements.txt requirements.txt
RUN uv pip install -r requirements.txt

# AWS Distro for OpenTelemetry — sends traces to CloudWatch (AgentCore Observability)
RUN uv pip install aws-opentelemetry-distro

# Run as a non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

# 8080 = HTTP agents, 8000 = MCP servers, 9000 = A2A agents
EXPOSE 9000
EXPOSE 8000
EXPOSE 8080

COPY . .

CMD ["opentelemetry-instrument", "python", "-m", "agent"]
