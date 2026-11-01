"""
trip_14_a2a_client_demo.py
──────────────────────────
Lesson 14 · A2A by hand: talk to the Local Guide agent yourself (like lesson 09 did for MCP).

The story
    Before Trip Buddy talks to the Local Guide agent, we do it ourselves in 2 steps:
      1. DISCOVER   read its Agent Card: who is it, what can it do, where is it?
      2. ASK        send it a message and print its reply

How to run
    Terminal 1:  uv run trip_guide_agent.py          (keep it running)
    Terminal 2:  uv run trip_14_a2a_client_demo.py

What you will see
    The agent card (name, skill, URL), then the guide's answer about Hampi in December.
"""

import asyncio

import httpx
from a2a.client import A2ACardResolver, A2AClientError, ClientConfig, create_client
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest

GUIDE_URL = "http://127.0.0.1:9999"


async def main() -> None:
    # 1. DISCOVER: download the agent card from /.well-known/agent-card.json
    async with httpx.AsyncClient() as http:
        card = await A2ACardResolver(httpx_client=http, base_url=GUIDE_URL).get_agent_card()
    print("── 1. AGENT CARD (discovery) " + "─" * 40)
    print("name       :", card.name)
    print("description:", card.description)
    for s in card.skills:
        print(f"skill      : {s.name} ({s.id}) - {s.description}")
        print("example    :", list(s.examples))
    print("reach it at:", card.supported_interfaces[0].url, "using", card.supported_interfaces[0].protocol_binding)

    # 2. ASK: send a message, get the reply
    question = "What should 4 friends do in Hampi in December?"
    print("\n── 2. SEND A MESSAGE " + "─" * 48)
    print("we ask     :", question)
    client = await create_client(card, ClientConfig(streaming=False))
    request = SendMessageRequest(message=new_text_message(question, role=Role.ROLE_USER))
    async for response in client.send_message(request):
        print("guide says :", get_stream_response_text(response))
    await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (A2AClientError, httpx.HTTPError):
        print("Can't reach the Local Guide agent. Start it first in another terminal:  uv run trip_guide_agent.py")
