"""
A Strands agent connected to the AWS Documentation MCP server.

The AWS Labs documentation server (https://github.com/awslabs/mcp) lets the
agent search and read official AWS docs, so its answers come from current
documentation instead of only the model's training data.

Prerequisite: `uv` installed (it provides the `uvx` command that downloads and
runs the server on demand).

Run:  python agent_with_aws_docs_mcp.py
"""

import os
import shutil

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models.bedrock import BedrockModel
from strands.tools.mcp import MCPClient

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

if shutil.which("uvx") is None:
    raise SystemExit("`uvx` not found. Install uv first: pip install uv")

# Create MCP client
mcp_client = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["awslabs.aws-documentation-mcp-server@latest"],
            env={**os.environ, "FASTMCP_LOG_LEVEL": "ERROR"},
        )
    ),
    startup_timeout=120,  # first run downloads the server package
)

# Use with agent
with mcp_client:
    tools = mcp_client.list_tools_sync()
    print("Tools from the AWS Documentation MCP server:", [t.tool_name for t in tools])

    agent = Agent(
        model=BedrockModel(model_id=MODEL_ID, region_name=REGION),
        tools=tools,
        system_prompt="Answer AWS questions using the documentation tools. Cite the doc URLs you used.",
    )
    agent("What is AWS Lambda, and what is the maximum timeout for a Lambda function?")
    print()
