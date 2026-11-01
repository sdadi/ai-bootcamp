"""
trip_12_memory_pipeline.py
──────────────────────────
Lesson 12 · The memory pipeline: short-term → long-term, watched step by step.

The story
    SESSION 1 (Monday): Riya chats with Trip Buddy. She mentions the group size,
    the budget, last March's Gokarna trip, that Aman is vegetarian, and says
    "always add a 10% buffer". All of it sits in short-term memory (the chat).

    The chat ends. Before the notepad is thrown away, the PIPELINE runs:
        1. EXTRACT      the LLM reads the chat and picks what is worth remembering
        2. CLASSIFY     it tags each item: semantic / episodic / procedural
        3. CONSOLIDATE  compare with the filing cabinet: ADD / UPDATE / SKIP
        4. STORE        file each item in the right drawer (SQLite or ChromaDB)

    SESSION 2 (Friday): a brand-new chat, so short-term memory is empty.
    Riya only says "Plan a 2-night trip for our gang next month."
        5. RECALL       Trip Buddy loads the long-term memories into its prompt
    ...and answers as if it knew them all along.

What you learn
    - Why long-term memory needs a pipeline (not everything in a chat is worth keeping)
    - Structured output: the LLM fills a form (MemoryList) instead of free text
    - Consolidation: a changed fact is UPDATED, a repeat is SKIPPED

How to run
    uv run trip_12_memory_pipeline.py            (needs OPENAI_API_KEY in .env)
    Run it a SECOND time: the pipeline will SKIP what it already knows.
    Start fresh:  uv run trip_12_memory_pipeline.py --reset
    Memories are saved in the memory_store/ folder (lesson 13 uses the same folder).
"""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from trip_memory import LongTermMemory, save_session, trim

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
llm = ChatOpenAI(model=MODEL, reasoning_effort="none")   # see lesson 04

ltm = LongTermMemory()            # the real filing cabinet: memory_store/
if "--reset" in sys.argv:
    ltm.forget_everything()
    print("Long-term memory cleared.")


def title(text: str) -> None:
    print(f"\n{'═' * 70}\n{text}\n{'═' * 70}")


def chat_turn(messages: list[dict], user_text: str, memory_text: str) -> str:
    """One chat turn. The LLM sees: system prompt (with long-term memory) + short-term memory."""
    messages.append({"role": "user", "content": user_text})
    lc = [SystemMessage(content="You are Trip Buddy, a friendly trip planner for groups of friends. "
                                "Keep answers to 2-3 sentences.\n\nLong-term memory:\n" + memory_text)]
    for m in trim(messages):                                  # short-term memory, trimmed
        lc.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
    reply = llm.invoke(lc).text
    messages.append({"role": "assistant", "content": reply})
    return reply


# =============================================================================
# SESSION 1: Monday's chat (short-term memory fills up)
# =============================================================================
title("SESSION 1 · Monday · short-term memory fills up")
monday = []                                                   # short-term memory = this list
for text in [
    "Hi! We're 4 friends from Bengaluru and our budget is usually ₹12,000 each.",
    "Last March we did Gokarna for 2 nights. We loved Om Beach, but the overnight bus made Neha sick.",
    "Also, Aman is vegetarian, so food options matter.",
    "From now on, always add a 10% buffer when you plan for us. We always overspend!",
]:
    print(f"\nRiya:       {text}")
    print(f"Trip Buddy: {chat_turn(monday, text, ltm.recall(text))}")

print(f"\nShort-term memory now holds {len(monday)} messages. The chat is ending...")

# =============================================================================
# THE PIPELINE: short-term → long-term
# =============================================================================
title("PIPELINE · steps 1-2 · EXTRACT and CLASSIFY (the LLM fills a MemoryList form)")
items, report = save_session(monday, ltm, llm)
for m in items:
    key = f" [{m.key}]" if m.type == "semantic" else ""
    print(f"  {m.type:<10}{key} {m.content}")
if not items:
    print("  (nothing worth remembering)")

title("PIPELINE · steps 3-4 · CONSOLIDATE and STORE")
for r in report:
    extra = f"   (was: {r['old']})" if r.get("old") else ""
    print(f"  {r['action']:<6} {r['type']:<10} → {r['text']}{extra}")
print("\nADD = new, UPDATE = changed fact, SKIP = already known")

monday = []                                                   # the chat ends: short-term memory is gone
print("\nSession 1 is over: messages = []. Only the filing cabinet remains.")

# =============================================================================
# SESSION 2: Friday, a brand-new chat
# =============================================================================
title("SESSION 2 · Friday · new chat, short-term memory is EMPTY")
friday = []
question = "Plan a 2-night trip for our gang next month."
memory_text = ltm.recall(question)
print("Step 5 · RECALL · this is put into the system prompt:\n")
print(memory_text)

print(f"\nRiya:       {question}")
print(f"Trip Buddy: {chat_turn(friday, question, memory_text)}")
print("\nRiya never repeated the group size, budget, food needs or the 10% rule. Trip Buddy remembered.")
