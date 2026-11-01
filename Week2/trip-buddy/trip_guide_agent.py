"""
trip_guide_agent.py
───────────────────
The Local Guide agent: a SEPARATE agent that Trip Buddy can talk to using A2A.

The story
    Trip Buddy is great at prices: rooms, travel and cost per person. But ask
    "What should we DO in Hampi?" and it has no idea. Another team has built a
    Local Guide agent that knows things to do, local food and weather for each place.
    Instead of copying their code, Trip Buddy simply ASKS the Local Guide agent,
    one agent talking to another. That's A2A (Agent-to-Agent).

    MCP vs A2A in one line:
        MCP = an agent uses TOOLS        (Trip Buddy → trip_mcp_server.py)
        A2A = an agent asks another AGENT (Trip Buddy → this Local Guide agent)

What this file has (the 3 parts of every A2A agent)
    1. AGENT CARD    its public profile: name, what it can do (skills), where to reach it.
                     Anyone can read it at http://127.0.0.1:9999/.well-known/agent-card.json
    2. THE BRAIN     a small notes dictionary + an LLM that writes a friendly answer
    3. EXECUTOR      receives a message, runs the brain, sends the reply back

How to run
    Unlike the MCP server, an A2A agent is a small web service that runs on its own.
    Open a SEPARATE terminal and keep it running:
        uv run trip_guide_agent.py            (needs OPENAI_API_KEY in .env)
    Then, in another terminal, run lesson 14 or 15. Stop it with Ctrl+C.
"""

import os

import uvicorn
from a2a.helpers import get_message_text, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from starlette.applications import Starlette

load_dotenv()

HOST, PORT = "127.0.0.1", 9999

# =============================================================================
# 1. AGENT CARD: the agent's public profile (how other agents discover it)
# =============================================================================
skill = AgentSkill(
    id="local_tips",
    name="Local tips",
    description="Things to do, local food to try and weather warnings for an Indian travel "
                "destination in a given month.",
    tags=["travel", "local guide"],
    examples=["What should we do in Hampi in December?"],
)

agent_card = AgentCard(
    name="Local Guide Agent",
    description="Knows what to do, what to eat and what weather to expect at popular trips from Bengaluru.",
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=f"http://{HOST}:{PORT}", protocol_version="1.0")],
    skills=[skill],
)

# =============================================================================
# 2. THE BRAIN: the guide's own notes + an LLM to write the answer
#    (Trip Buddy does NOT have these notes: that's why it asks this agent)
# =============================================================================
LOCAL_NOTES = {
    "goa":         "Do: Fort Aguada at sunset, Fontainhas walk in Panaji, Dudhsagar falls. Eat: fish thali, bebinca (veg: mushroom xacuti).",
    "gokarna":     "Do: Om Beach, the cliff trek from Kudle to Half Moon Beach, Mahabaleshwar temple. Eat: Udupi thali, neer dosa.",
    "pondicherry": "Do: White Town walk, Auroville, Paradise Beach. Eat: French bakery croissants, Creole curry.",
    "coorg":       "Do: Abbey Falls, a coffee plantation tour, Raja's Seat at sunset. Eat: akki roti, bamboo shoot curry.",
    "ooty":        "Do: toy train to Coonoor, Botanical Garden, Doddabetta Peak. Eat: homemade chocolates, varkey biscuits.",
    "manali":      "Do: Solang Valley, Old Manali cafes, Hadimba Temple. Eat: siddu, rajma chawal.",
    "hampi":       "Do: Virupaksha Temple, sunrise at Matanga Hill, coracle ride on the Tungabhadra. Eat: thali meals, banana pancakes.",
    "jaipur":      "Do: Amber Fort, Hawa Mahal, Nahargarh at sunset. Eat: dal baati churma, pyaaz kachori.",
    "rishikesh":   "Do: river rafting (Oct–Jun), Ganga aarti at Triveni Ghat, Laxman Jhula. Eat: Chotiwala thali, cafe pancakes.",
    "dandeli":     "Do: white-water rafting on the Kali (Oct–May), Syntheri Rocks, jungle safari. Eat: Malnad meals, kokum juice.",
}

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-6-luna"), reasoning_effort="none")   # see lesson 04


async def guide_brain(question: str) -> str:
    """Find our notes for the place in the question, then let the LLM write a short answer."""
    notes = next((n for place, n in LOCAL_NOTES.items() if place in question.lower()), "No notes for this place.")
    reply = await llm.ainvoke([
        ("system", "You are a friendly local guide for Indian destinations. Using the notes, give 3 things to do, "
                   "1 local food to try, and 1 weather warning for the month asked. Under 80 words.\n"
                   f"Notes: {notes}"),
        ("human", question),
    ])
    return reply.text


# =============================================================================
# 3. EXECUTOR: receives a message → runs the brain → sends the reply back
# =============================================================================
class GuideExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = get_message_text(context.message)              # the text another agent sent us
        print(f"[guide] asked: {question}")
        answer = await guide_brain(question)
        await event_queue.enqueue_event(new_text_message(answer, role=Role.ROLE_AGENT))   # reply

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel is not supported.")


# =============================================================================
# Start the web service: agent card at /.well-known/agent-card.json, messages at /
# =============================================================================
handler = DefaultRequestHandler(agent_executor=GuideExecutor(), task_store=InMemoryTaskStore(), agent_card=agent_card)
app = Starlette(routes=create_agent_card_routes(agent_card) + create_jsonrpc_routes(handler, "/"))

if __name__ == "__main__":
    print(f"Local Guide Agent running at http://{HOST}:{PORT}")
    print(f"Agent card: http://{HOST}:{PORT}/.well-known/agent-card.json")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
