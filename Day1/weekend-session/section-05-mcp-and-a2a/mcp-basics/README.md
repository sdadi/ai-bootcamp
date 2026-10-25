# MCP Basics — Server and Client

A [Model Context Protocol](https://modelcontextprotocol.io/) server and client over stdio. No LLM is needed — this is the protocol itself.

```
┌────────────┐  stdio (stdin/stdout)  ┌────────────┐
│  client.py │ ◄────────────────────► │  server.py │
│ MCP Client │      JSON-RPC          │ MCP Server │
└────────────┘                        └────────────┘
```

The client starts the server as a subprocess; they exchange JSON-RPC messages through pipes. No ports, no HTTP.

## What the server exposes

| Primitive | Name | What it is |
|---|---|---|
| Tool | `get_weather(city)` | A function the LLM can call |
| Tool | `list_client_roots()` | Asks the *client* which folders it shared (roots) |
| Resource (template) | `greeting://{name}` | Data the app can read, like a GET endpoint |
| Resource (JSON) | `courses://directory` | A fixed resource with `application/json` content |
| Prompt | `weather_report(city)` | A reusable prompt template |
| Prompt | `generate_study_plan(course, exam_date)` | A template with two arguments |

And the client exposes **roots** — the folders the server is allowed to work in.

## Run

```bash
pip install -r requirements.txt
python client.py        # starts server.py automatically
```

Expected output (abridged):

```
=== Available Tools ===
  - get_weather: Get the current weather for a given city.
  - list_client_roots: List the workspace roots the connected client has shared with this server.
=== Available Resources ===
  - courses://directory (application/json): List of courses and instructors in a program
  - greeting://{name}: Return a personalized greeting.
=== Available Prompts ===
  - weather_report(city): ...
  - generate_study_plan(course, exam_date): ...
=== Calling get_weather('London') ===
  Result: Weather in London: 15°C, cloudy
...
=== Calling list_client_roots ===
  Result: Course workspace: file:///.../mcp-basics
```

Want a visual tool? Run `npx @modelcontextprotocol/inspector` and connect it to `python server.py`.
