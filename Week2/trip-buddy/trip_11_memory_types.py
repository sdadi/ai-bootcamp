"""
trip_11_memory_types.py
───────────────────────
Lesson 11 · The 4 kinds of memory, with Trip Buddy examples (no AI yet).

The story
    Riya's gang uses Trip Buddy every few months. A good human travel agent would
    remember them: their group size, that Aman is vegetarian, that the Gokarna trip
    was great, and that they always overspend. Lesson 08's Trip Buddy forgets
    everything the moment the chat ends. Let's look at each kind of memory it needs.

    SHORT-TERM  the notepad during today's call     the current chat's messages
    SEMANTIC    the customer profile card           facts: "group_size: 4 friends"
    EPISODIC    the trip history folder             past trips: "March 2026: Gokarna..."
    PROCEDURAL  the "how I serve them" card         rules: "always add a 10% buffer"

What you learn
    - What each memory type is for, with a use case for each
    - Short-term memory is a Python list: it has a size limit and dies with the chat
    - Long-term memory is saved on disk (SQLite + ChromaDB) and survives restarts
    - RECALL: how long-term memory is turned into text for the system prompt

How to run
    uv run trip_11_memory_types.py
    (no API key needed). Then RUN IT A SECOND TIME: step 0 will show the memories
    saved by the first run, which proves long-term memory survives a restart.
    Demo memories go in a separate folder, memory_store_demo/.
"""

from trip_memory import LongTermMemory, trim


def title(text: str) -> None:
    print(f"\n{'═' * 70}\n{text}\n{'═' * 70}")


ltm = LongTermMemory(folder="memory_store_demo")   # demo filing cabinet (separate from lessons 12-13)

# ── 0. Is anything left from the last run? ────────────────────────────────────
title("0. LONG-TERM MEMORY SURVIVES RESTARTS")
facts = ltm.all_facts()
if facts:
    print("Found on disk from the last run (nothing was typed this time):")
    for k, v in facts.items():
        print(f"  {k}: {v}")
else:
    print("Nothing on disk yet. Run this file again after it finishes and you'll see the facts here.")
ltm.forget_everything()   # start the demo clean each time

# ── 1. SHORT-TERM memory: the current chat ────────────────────────────────────
title("1. SHORT-TERM MEMORY  (the notepad during today's call)")
messages = [
    {"role": "user",      "content": "Can 4 of us do Goa for 3 nights on ₹12,000 each?"},
    {"role": "assistant", "content": "Yes! ₹11,800 each, ₹200 left."},
    {"role": "user",      "content": "And if we are 5 instead?"},     # ← makes sense ONLY with the turn above
]
print("Use case: follow-up questions. 'And if we are 5 instead?' only makes sense")
print("because the earlier messages (Goa, 3 nights, ₹12,000) are still in the list:\n")
for m in messages:
    print(f"  {m['role']:>9}: {m['content']}")

long_chat = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i + 1}"} for i in range(14)]
kept = trim(long_chat, keep_last=6)
print(f"\nLimit: the LLM can only read so much (context window). A chat with {len(long_chat)} messages")
print(f"is trimmed to the last {len(kept)}: {[m['content'] for m in kept]}")
print("Messages 1-8 fell out of short-term memory.")

messages = []   # the chat ends
print("\nChat ends → messages = []  Short-term memory is gone. Without long-term memory, everything is forgotten.")

# ── 2. SEMANTIC memory: facts about the group ─────────────────────────────────
title("2. SEMANTIC MEMORY  (the customer profile card)")
print("Use case: no need to repeat yourself. Trip Buddy already knows your group.\n")
for key, value in [("group_size", "4 friends"), ("home_city", "Bengaluru"),
                   ("budget_per_person", "₹12,000 each"), ("food", "Aman is vegetarian")]:
    r = ltm.set_fact(key, value)
    print(f"  {r['action']:<6} {r['text']}")
print("\nFacts change. One key holds one value, so a new value replaces the old one:")
r = ltm.set_fact("budget_per_person", "₹15,000 each")
print(f"  {r['action']:<6} {r['text']}   (was: {r['old']})")
r = ltm.set_fact("home_city", "Bengaluru")
print(f"  {r['action']:<6} {r['text']}   (already known, nothing to do)")

# ── 3. EPISODIC memory: past trips ────────────────────────────────────────────
title("3. EPISODIC MEMORY  (the trip history folder)")
print("Use case: learn from experience: repeat what went well, avoid what went badly.\n")
for ep in [
    "March 2026: Gokarna, 4 friends, 2 nights. Loved Om Beach, but the overnight bus made Neha sick.",
    "December 2025: Goa over New Year. Very crowded and prices were double. Not again at New Year.",
    "October 2025: Coorg, 3 friends, 2 nights. Coffee estate stay was the highlight.",
]:
    print(f"  {ltm.add_episode(ep)['action']:<6} {ep}")
print("\nEpisodes are searched by MEANING (ChromaDB, like lesson 03):")
for q in ["a beach trip we enjoyed", "a trip that went badly"]:
    print(f"  search '{q}' → {ltm.search_episodes(q, n=1)[0]}")

# ── 4. PROCEDURAL memory: how to plan for this group ──────────────────────────
title("4. PROCEDURAL MEMORY  (the 'how I serve this group' card)")
print("Use case: the group taught Trip Buddy HOW to work for them. These become instructions.\n")
for rule in ["Always add a 10% buffer to the total, this group overspends.",
             "Avoid overnight buses; suggest trains instead.",
             "Always add a 10% buffer to the total, this group overspends."]:   # a repeat → SKIP
    r = ltm.add_rule(rule)
    print(f"  {r['action']:<6} {rule}")

# ── 5. RECALL: long-term memory → system prompt ───────────────────────────────
title("5. RECALL  (what an agent puts in its system prompt for a new question)")
question = "Plan a beach weekend for us next month."
print(f"New chat, short-term memory is empty. Question: '{question}'\n")
print(ltm.recall(question))
print("\nNotice: all facts and rules are included; only the past trips CLOSEST to the question are picked.")
