"""
MCP Server — exposes all the MCP primitives over stdio transport:

  * Tools      — get_weather, list_client_roots
  * Resources  — greeting://{name}  and  courses://directory (JSON)
  * Prompts    — weather_report, generate_study_plan

MCP (Model Context Protocol) lets LLM applications discover and call tools,
read resources, and use prompt templates through one standard protocol.

Run:  python client.py   (the client starts this server for you)
"""

import json

# FastMCP: high-level helper for building MCP servers. It handles JSON-RPC
# parsing, registration of tools/resources/prompts, and the transport.
from mcp.server.fastmcp import Context, FastMCP

# The name identifies this server to clients when they connect.
mcp = FastMCP("WeatherServer", log_level="WARNING")


# --- Tool: get_weather ---------------------------------------------------
# Tools are functions an LLM can call. @mcp.tool() registers the function so
# it appears in list_tools() and can be invoked with call_tool(). The
# docstring becomes the description the LLM sees.

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    # Fake data for demo purposes — a real server would call a weather API
    fake_weather = {
        "london": "15°C, cloudy",
        "paris": "18°C, sunny",
        "tokyo": "22°C, humid",
        "new york": "12°C, windy",
        "bangalore": "26°C, pleasant with light showers",
    }
    result = fake_weather.get(city.lower())
    if result:
        return f"Weather in {city}: {result}"
    return f"Sorry, no weather data available for '{city}'."


# --- Tool that uses a client capability: roots ---------------------------
# Roots are folders/URIs the *client* says the server may work within.
# A server can ask the client for its roots at any time.

@mcp.tool()
async def list_client_roots(ctx: Context) -> str:
    """List the workspace roots the connected client has shared with this server."""
    result = await ctx.session.list_roots()
    if not result.roots:
        return "The client did not share any roots."
    return "\n".join(f"{root.name}: {root.uri}" for root in result.roots)


# --- Resource: greeting ---------------------------------------------------
# Resources are data an LLM application can read (like GET endpoints).
# "greeting://{name}" is a URI template: read_resource("greeting://Developer")
# fills in {name}. Use resources for data, tools for actions.

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Return a personalized greeting."""
    return f"Hello, {name}! Welcome to the WeatherServer."


# --- Resource: course directory (structured JSON) -------------------------
# A fixed URI with a MIME type tells the client how to interpret the content.

COURSES = [
    {"course": "Machine Learning", "instructor": "Dr. Rao", "weeks": 12},
    {"course": "Agentic AI on AWS", "instructor": "Shivank Agrawal", "weeks": 8},
    {"course": "Data Engineering", "instructor": "Ms. Iyer", "weeks": 10},
]


@mcp.resource(
    "courses://directory",
    name="courseDirectory",
    title="Course Directory",
    description="List of courses and instructors in a program",
    mime_type="application/json",
)
def course_directory() -> str:
    return json.dumps(COURSES, indent=2)


# --- Prompts ---------------------------------------------------------------
# Prompts are reusable templates. Clients call get_prompt(name, arguments)
# and get back ready-to-send messages.

@mcp.prompt()
def weather_report(city: str) -> str:
    """Generate a prompt asking for a weather report."""
    return f"Please provide a detailed weather report for {city}, including temperature, humidity, and forecast."


@mcp.prompt(title="Generate Study Plan")
def generate_study_plan(course: str, exam_date: str) -> str:
    """Create a personalized study plan for a student based on their course and exam date."""
    return (
        f"Create a week-by-week study plan for the course '{course}'. "
        f"The exam is on {exam_date}. Include revision weeks, practice tests, "
        "and one concrete goal per week."
    )


# --- Start the server ------------------------------------------------------
# "stdio" = read JSON-RPC messages from stdin, write responses to stdout.
# The client spawns this script as a subprocess and talks to it through pipes.

if __name__ == "__main__":
    mcp.run(transport="stdio")
