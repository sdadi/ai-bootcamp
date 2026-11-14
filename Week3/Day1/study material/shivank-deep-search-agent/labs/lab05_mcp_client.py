"""
LAB 5b — An MCP client that starts the server, discovers its capabilities and calls tools
================================================================================
Course module: 4 (MCP)

What you will see
  1. The client launches lab05_mcp_server.py as a subprocess and talks to it over stdio.
     (Older tutorials use stdio_client + ClientSession + session.initialize(); that still works in
     SDK v2, but the high-level Client below is simpler and handles version negotiation for you.)
  2. Discovery: list_tools / list_resources / list_prompts — the agent learns what exists at runtime.
  3. Calling tools by name with JSON arguments, exactly as an LLM-driven agent would.

Run it
  python labs/lab05_mcp_client.py

Use the server from Claude Desktop (optional)
  Add this to claude_desktop_config.json (use YOUR absolute paths), then restart Claude Desktop:
  {
    "mcpServers": {
      "shivank-deep-search-tools": {
        "command": "/ABSOLUTE/PATH/TO/.venv/bin/python",
        "args": ["/ABSOLUTE/PATH/TO/shivank-deep-search-agent/labs/lab05_mcp_server.py"]
      }
    }
  }
"""

from _common import ROOT, banner, pause_note, step  # noqa: E402

import asyncio
import json
import sys

from mcp import Client, StdioServerParameters

SERVER = ROOT / "labs" / "lab05_mcp_server.py"


def _json_blocks(result) -> list:
    """A tool result is a list of content blocks. Our tools return JSON, one block per item.
    (Newer servers may also send `structured_content`, a ready-made dict — we use it when present.)"""
    if getattr(result, "structured_content", None) and "result" in result.structured_content:
        return result.structured_content["result"]
    return [json.loads(block.text) for block in result.content if getattr(block, "text", "")]


async def run() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(ROOT))
    # Client starts the server process, negotiates the protocol version and keeps the connection open.
    async with Client(params) as session:
        print(f"  connected — negotiated MCP protocol version {session.protocol_version}")

        step("1. Discovery: what does this server offer?")
        tools = await session.list_tools()
        for t in tools.tools:
            print(f"  tool     {t.name:<8} — {t.description.splitlines()[0]}")
            print(f"           input schema: {json.dumps(t.input_schema)[:120]}…")
        resources = await session.list_resources()
        for r in resources.resources:
            print(f"  resource {r.uri}")
        prompts = await session.list_prompts()
        for p in prompts.prompts:
            print(f"  prompt   {p.name} (args: {[a.name for a in (p.arguments or [])]})")

        step("2. Call the 'search' tool")
        res = await session.call_tool("search", {"query": "NimbusPay payment aggregator registry", "max_results": 3})
        for row in _json_blocks(res):
            print(f"  #{row['rank']} {row['url']}  ({row['published_date']})")

        step("3. Call the 'fetch' tool")
        res = await session.call_tool("fetch", {"url": "https://registry.regulator.example/payment-aggregators/nimbuspay"})
        page = _json_blocks(res)[0]
        print(f"  {page['meta']['title']}  [{page['meta']['source_type']}]")
        print(f"  {page['text'][:160]}…")

        step("4. Read the resource and render the prompt")
        body = await session.read_resource("corpus://sources")
        print("  corpus://sources →", len(json.loads(body.contents[0].text)), "pages listed")
        prompt = await session.get_prompt("research_brief", {"company": "NimbusPay", "city": "Bengaluru"})
        print("  research_brief →", prompt.messages[0].content.text[:110], "…")


def main() -> None:
    banner("LAB 5 — MCP client ↔ MCP server")
    asyncio.run(run())
    pause_note("Tools = what the MODEL can call. Resources = context the APP can attach. Prompts = templates "
               "the USER picks. One server, reusable by every MCP-compatible client and agent.")


if __name__ == "__main__":
    main()
