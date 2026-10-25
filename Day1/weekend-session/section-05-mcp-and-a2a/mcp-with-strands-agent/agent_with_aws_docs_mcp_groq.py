"""
A Strands agent connected to the AWS Documentation MCP server, powered by Groq.

Same as agent_with_aws_docs_mcp.py; only the model provider changes. No AWS
credentials are needed: the docs server reads public documentation.

Prerequisite: `uv` installed (it provides the `uvx` command).

Run:  python agent_with_aws_docs_mcp_groq.py    (needs GROQ_API_KEY, or a .env file)
"""

import os
import shutil

from dotenv import load_dotenv
from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient

load_dotenv()

MODEL_ID = os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-120b")

if shutil.which("uvx") is None:
    raise SystemExit("`uvx` not found. Install uv first: pip install uv")

model = OpenAIModel(
    client_args={
        "api_key": os.environ["GROQ_API_KEY"],
        "base_url": os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),  # or a local OpenAI-compatible server
        "max_retries": 5,  # retry dropped connections (the client default is 2)
    },
    model_id=MODEL_ID,
    # Groq reasoning models stream their thinking by default; hide it so only the answer prints.
    params={"extra_body": {"include_reasoning": False}},
)

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
        model=model,
        tools=tools,
        system_prompt="Answer AWS questions using the documentation tools. Cite the doc URLs you used.",
    )
    agent("What is AWS Lambda, and what is the maximum timeout for a Lambda function?")
    print()
