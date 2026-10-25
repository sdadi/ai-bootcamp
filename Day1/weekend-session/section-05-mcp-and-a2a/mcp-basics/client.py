"""
MCP Client — starts server.py over stdio and exercises every MCP primitive:
tools, resources, prompts, and roots.

Run:  python client.py
"""

import asyncio
import json
import sys
from pathlib import Path

import mcp.types as types

# ClientSession: the MCP protocol session (initialize, list, call, ...)
# StdioServerParameters: how to start the server subprocess
from mcp import ClientSession, StdioServerParameters

# stdio_client: starts the server process and returns read/write streams
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent


# --- Roots -----------------------------------------------------------------
# The client decides which folders the server may work within. When the
# server asks (list_roots), this callback answers.

async def list_roots(context) -> types.ListRootsResult:
    return types.ListRootsResult(
        roots=[
            types.Root(uri=HERE.as_uri(), name="Course workspace"),
        ]
    )


async def main():
    # Start "python server.py" as a subprocess. sys.executable is the same
    # Python that runs this client, so the server sees the same packages.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(HERE / "server.py")],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream, list_roots_callback=list_roots) as session:
            # Required first step: exchange protocol version and capabilities
            await session.initialize()

            # --- 1. Discover tools ----------------------------------------
            tools = await session.list_tools()
            print("=== Available Tools ===")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # --- 2. Discover resources ------------------------------------
            # Fixed resources and URI templates are listed separately.
            print("\n=== Available Resources ===")
            resources = await session.list_resources()
            for r in resources.resources:
                print(f"  - {r.uri} ({r.mimeType}): {r.description}")
            templates = await session.list_resource_templates()
            for rt in templates.resourceTemplates:
                print(f"  - {rt.uriTemplate}: {rt.description}")

            # --- 3. Discover prompts --------------------------------------
            print("\n=== Available Prompts ===")
            prompts = await session.list_prompts()
            for p in prompts.prompts:
                args = ", ".join(a.name for a in (p.arguments or []))
                print(f"  - {p.name}({args}): {p.description}")

            # --- 4. Call a tool (known city) ------------------------------
            print("\n=== Calling get_weather('London') ===")
            result = await session.call_tool("get_weather", {"city": "London"})
            print(f"  Result: {result.content[0].text}")

            # --- 5. Call a tool (unknown city) ----------------------------
            print("\n=== Calling get_weather('Berlin') ===")
            result = await session.call_tool("get_weather", {"city": "Berlin"})
            print(f"  Result: {result.content[0].text}")

            # --- 6. Read a templated resource -----------------------------
            print("\n=== Reading greeting://Developer ===")
            resource = await session.read_resource("greeting://Developer")
            print(f"  Result: {resource.contents[0].text}")

            # --- 7. Read a JSON resource ----------------------------------
            print("\n=== Reading courses://directory ===")
            resource = await session.read_resource("courses://directory")
            courses = json.loads(resource.contents[0].text)
            for c in courses:
                print(f"  - {c['course']} — {c['instructor']} ({c['weeks']} weeks)")

            # --- 8. Get prompts -------------------------------------------
            print("\n=== Getting weather_report prompt ===")
            prompt = await session.get_prompt("weather_report", {"city": "Tokyo"})
            print(f"  Result: {prompt.messages[0].content.text}")

            print("\n=== Getting generate_study_plan prompt ===")
            prompt = await session.get_prompt(
                "generate_study_plan", {"course": "Machine Learning", "exam_date": "2026-05-10"}
            )
            print(f"  Result: {prompt.messages[0].content.text}")

            # --- 9. Roots: the server asks the client ---------------------
            print("\n=== Calling list_client_roots ===")
            result = await session.call_tool("list_client_roots", {})
            print(f"  Result: {result.content[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
