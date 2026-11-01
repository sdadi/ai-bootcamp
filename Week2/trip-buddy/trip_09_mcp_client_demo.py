"""
trip_09_mcp_client_demo.py
──────────────────────────
Lesson 09 · MCP client: talk to the Trip Buddy MCP server by hand (no AI yet).

The story
    Before an AI uses our MCP server, we act as the client ourselves and walk
    through every step of the protocol, one at a time:
      1. start the server            stdio_client launches trip_mcp_server.py
      2. handshake                   session.initialize()
      3. tool discovery              session.list_tools()
      4. tool invocation             session.call_tool(...)
      5. a validation error          isError = True (people = 0 is not allowed)
      6. resources                   list_resources() / read_resource(uri)
      7. prompts                     list_prompts() / get_prompt(name, args)

What you learn
    - Host, client and server roles: this file is the host, ClientSession is the client
    - MCP calls are async, so we use async/await and asyncio.run(...)
    - The JSON Schema the LLM will see for each tool (built by FastMCP from type hints)

How to run
    uv run trip_09_mcp_client_demo.py
    (no API key needed: there is no LLM in this lesson)

What you will see
    A printed walkthrough of all 7 steps with the server's real replies.
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# How to start the server: "run trip_mcp_server.py with the same Python as this file".
SERVER = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).parent / "trip_mcp_server.py")],
)


def title(text: str) -> None:
    print(f"\n{'─' * 70}\n{text}\n{'─' * 70}")


async def main() -> None:
    # 1. Start the server as a subprocess and connect to its stdin/stdout
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:

            # 2. Handshake: must happen before any other call
            title("2. HANDSHAKE  session.initialize()")
            init = await session.initialize()
            print("server name     :", init.serverInfo.name)
            print("protocol version:", init.protocolVersion)
            caps = init.capabilities
            print("server offers   :", [c for c in ("tools", "resources", "prompts") if getattr(caps, c)])

            # 3. Tool discovery: this is exactly what the LLM will be shown
            title("3. TOOL DISCOVERY  session.list_tools()")
            tools = (await session.list_tools()).tools
            for t in tools:
                print(f"\n• {t.name}")
                print("  description:", (t.description or "").strip().splitlines()[0])
                print("  inputs     :", json.dumps(t.inputSchema.get("properties", {}), ensure_ascii=False)[:160], "...")

            # 4. Tool invocation
            title("4. TOOL CALL  session.call_tool('compute_trip_cost', {...})")
            args = {"travel_cost_per_person": 2800, "room_price_per_night": 3000,
                    "daily_spend_per_person": 1500, "people": 4, "nights": 3,
                    "budget_per_person": 12000}
            print("arguments:", args)
            result = await session.call_tool("compute_trip_cost", args)
            print("isError  :", result.isError)
            print("result   :", result.content[0].text)

            # 5. A validation error: people must be between 1 and 20
            title("5. VALIDATION ERROR  people = 0")
            bad = await session.call_tool("compute_trip_cost", {**args, "people": 0})
            print("isError  :", bad.isError)
            print("message  :", bad.content[0].text[:200])

            # 6. Resources: read-only data at a URI
            title("6. RESOURCES  list_resources() / read_resource(uri)")
            for r in (await session.list_resources()).resources:
                print("resource         :", r.uri, "-", r.description)
            for rt in (await session.list_resource_templates()).resourceTemplates:
                print("resource template:", rt.uriTemplate, "-", rt.description)
            places = json.loads((await session.read_resource("trips://destinations")).contents[0].text)
            print(f"\ntrips://destinations → {len(places)} places, first one:", places[0])
            hampi = (await session.read_resource("trips://place/HAMPI")).contents[0].text
            print("trips://place/HAMPI  →", hampi)

            # 7. Prompts: ready-made messages the user picks
            title("7. PROMPTS  list_prompts() / get_prompt(name, args)")
            for p in (await session.list_prompts()).prompts:
                print("prompt:", p.name, [a.name for a in (p.arguments or [])])
            rendered = await session.get_prompt(
                "plan_trip", {"place": "Goa", "people": "4", "nights": "3", "budget_per_person": "12000"}
            )
            print("\nrendered message:", rendered.messages[0].content.text)

    print("\nDone. The server stopped when the session closed.")


if __name__ == "__main__":
    asyncio.run(main())
