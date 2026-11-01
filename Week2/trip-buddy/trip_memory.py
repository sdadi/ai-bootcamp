"""
trip_memory.py
──────────────
Trip Buddy's memory system: the "filing cabinet" (long-term memory) and the
pipeline that moves useful things from a chat (short-term memory) into it.

You don't run this file. Lessons 11, 12 and 13 import it.

The 4 kinds of memory (think of a good travel agent who knows you)
    SHORT-TERM  (working memory)  the notepad during today's phone call
        The messages of the current chat. Lets you say "make it 5 friends instead".
        Gone when the chat ends. Kept short so it fits in the LLM's context window.
    LONG-TERM, saved on disk, survives restarts:
    SEMANTIC    facts about you             your customer profile card
        "group_size: 4 friends", "food: Aman is vegetarian"       → SQLite table `facts`
    EPISODIC    past experiences            your trip history folder
        "March 2026: Gokarna, 2 nights, loved Om Beach"           → ChromaDB (search by meaning)
    PROCEDURAL  how to serve you            the agent's "how I handle this group" card
        "Always add a 10% buffer when planning for this group"    → SQLite table `rules`

    Not memory: the travel notebook (RAG) from lesson 05. The notebook is what Trip Buddy
    knows about the WORLD (same for everyone). Memory is what it knows about YOU.

The pipeline: short-term → long-term (runs when a chat ends)
    1. EXTRACT      an LLM reads the chat and lists what is worth remembering
    2. CLASSIFY     each item is tagged semantic / episodic / procedural
    3. CONSOLIDATE  compare with what we already know: ADD new, UPDATE changed, SKIP repeats
    4. STORE        write each item into the right drawer of the filing cabinet
    Next chat:
    5. RECALL       pull the relevant memories and put them in the system prompt
"""

import sqlite3
from datetime import date
from pathlib import Path
from typing import Literal

import chromadb
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# =============================================================================
# Short-term memory helpers
# =============================================================================
def text_of(content) -> str:
    """Chat UIs sometimes store a message as plain text, sometimes as a list of parts. Get the text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def trim(messages: list[dict], keep_last: int = 10) -> list[dict]:
    """Short-term memory has a size limit (the context window). Keep only the last few messages."""
    return messages[-keep_last:]


# =============================================================================
# Long-term memory: the filing cabinet on disk
# =============================================================================
class LongTermMemory:
    """Long-term memory saved in a folder next to this file. It survives restarts."""

    def __init__(self, folder: str = "memory_store", embed=None):
        self.path = Path(__file__).parent / folder
        self.path.mkdir(exist_ok=True)

        # The meaning machine (same model as lesson 02). Passed in, or created here.
        if embed is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                          model_kwargs={"device": "cpu"},
                                          encode_kwargs={"normalize_embeddings": True})
        self.embed = embed

        # SEMANTIC + PROCEDURAL drawers: a SQLite file (a real file this time, not :memory:)
        self.db = sqlite3.connect(self.path / "long_term.db", check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY, value TEXT, updated TEXT);
            CREATE TABLE IF NOT EXISTS rules (rule TEXT PRIMARY KEY, added TEXT);
        """)

        # EPISODIC drawer: ChromaDB saved to disk, so we can search past trips by meaning
        self.chroma = chromadb.PersistentClient(path=str(self.path / "episodes"))
        self.episodes = self.chroma.get_or_create_collection("episodes", metadata={"hnsw:space": "cosine"})

    # ── SEMANTIC: facts, one value per key ────────────────────────────────────
    def set_fact(self, key: str, value: str) -> dict:
        """Save a fact. Returns what happened: ADD (new), UPDATE (changed) or SKIP (same as before)."""
        key = key.strip().lower().replace(" ", "_") or "note"
        row = self.db.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
        if row and row[0].strip().lower() == value.strip().lower():
            return {"action": "SKIP", "type": "semantic", "text": f"{key}: {value}"}
        self.db.execute("INSERT OR REPLACE INTO facts VALUES (?, ?, ?)", (key, value, str(date.today())))
        self.db.commit()
        if row:
            return {"action": "UPDATE", "type": "semantic", "text": f"{key}: {value}", "old": row[0]}
        return {"action": "ADD", "type": "semantic", "text": f"{key}: {value}"}

    def all_facts(self) -> dict:
        return dict(self.db.execute("SELECT key, value FROM facts ORDER BY key").fetchall())

    # ── EPISODIC: past trips, searched by meaning ─────────────────────────────
    def add_episode(self, text: str) -> dict:
        """Save a past experience, unless an almost identical one is already saved."""
        vec = self.embed.embed_query(text)
        if self.episodes.count():
            near = self.episodes.query(query_embeddings=[vec], n_results=1)
            if near["distances"][0][0] < 0.08:            # 0 = identical meaning
                return {"action": "SKIP", "type": "episodic", "text": text}
        self.episodes.add(ids=[f"ep-{self.episodes.count() + 1}-{abs(hash(text)) % 10_000}"],
                          embeddings=[vec], documents=[text], metadatas=[{"saved": str(date.today())}])
        return {"action": "ADD", "type": "episodic", "text": text}

    def search_episodes(self, query: str, n: int = 2) -> list[str]:
        """The past trips closest in meaning to the question."""
        if not self.episodes.count():
            return []
        res = self.episodes.query(query_embeddings=[self.embed.embed_query(query)],
                                  n_results=min(n, self.episodes.count()))
        return res["documents"][0]

    def all_episodes(self) -> list[str]:
        return self.episodes.get()["documents"] if self.episodes.count() else []

    # ── PROCEDURAL: rules about HOW to plan for this group ────────────────────
    def add_rule(self, rule: str) -> dict:
        exists = self.db.execute("SELECT 1 FROM rules WHERE LOWER(rule) = LOWER(?)", (rule.strip(),)).fetchone()
        if exists:
            return {"action": "SKIP", "type": "procedural", "text": rule}
        self.db.execute("INSERT INTO rules VALUES (?, ?)", (rule.strip(), str(date.today())))
        self.db.commit()
        return {"action": "ADD", "type": "procedural", "text": rule}

    def all_rules(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT rule FROM rules ORDER BY added, rule").fetchall()]

    # ── RECALL: what goes into the system prompt for a new question ───────────
    def recall(self, question: str) -> str:
        facts, rules = self.all_facts(), self.all_rules()
        episodes = self.search_episodes(question, n=2)
        if not (facts or rules or episodes):
            return "Nothing saved yet. This is a new group."
        lines = []
        if facts:
            lines.append("Facts about this group (semantic):")
            lines += [f"- {k}: {v}" for k, v in facts.items()]
        if episodes:
            lines.append("Related past trips (episodic):")
            lines += [f"- {e}" for e in episodes]
        if rules:
            lines.append("How to plan for this group (procedural):")
            lines += [f"- {r}" for r in rules]
        return "\n".join(lines)

    def as_markdown(self) -> str:
        """Everything in the cabinet, for showing on screen."""
        f, e, r = self.all_facts(), self.all_episodes(), self.all_rules()
        out = ["**Semantic (facts)**"] + ([f"- `{k}`: {v}" for k, v in f.items()] or ["- (empty)"])
        out += ["", "**Episodic (past trips)**"] + ([f"- {x}" for x in e] or ["- (empty)"])
        out += ["", "**Procedural (rules)**"] + ([f"- {x}" for x in r] or ["- (empty)"])
        return "\n".join(out)

    def forget_everything(self) -> None:
        """Delete all long-term memory (a user should always be able to do this)."""
        self.db.executescript("DELETE FROM facts; DELETE FROM rules;")
        self.db.commit()
        self.chroma.delete_collection("episodes")
        self.episodes = self.chroma.get_or_create_collection("episodes", metadata={"hnsw:space": "cosine"})


# =============================================================================
# The pipeline: short-term → long-term
# =============================================================================
class MemoryItem(BaseModel):
    """One thing worth remembering. The LLM fills in this form."""
    type: Literal["semantic", "episodic", "procedural"] = Field(
        description="semantic = a lasting fact about the group; episodic = a past trip or event; "
                    "procedural = an instruction about HOW to plan for this group in future")
    key: str = Field(description="For semantic only: a short snake_case topic such as group_size, home_city, "
                                 "budget_per_person, food, health, likes, dislikes. Use an empty string otherwise.")
    content: str = Field(description="The memory itself, one short sentence.")


class MemoryList(BaseModel):
    memories: list[MemoryItem]


EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are the memory manager for Trip Buddy, a trip planner for a group of friends.\n"
     "Read the chat and pick out ONLY what will help plan FUTURE trips for this group:\n"
     "- semantic: lasting facts (group size, home city, usual budget, food needs, health, likes, dislikes)\n"
     "- episodic: a past trip or event, with when, where and how it went, in one sentence\n"
     "- procedural: an instruction about how Trip Buddy should plan for them from now on\n"
     "Skip small talk, one-off questions and Trip Buddy's own answers.\n"
     "Never save passwords, card numbers, phone numbers or ID numbers.\n"
     "Reuse the same key when a fact already exists (so it can be updated).\n"
     "Return an empty list if nothing is worth remembering."),
    ("human", "Already in long-term memory:\n{existing}\n\nChat to read:\n{transcript}"),
])


def transcript_of(messages: list[dict]) -> str:
    """Turn the short-term memory (list of messages) into plain text for the extractor."""
    who = {"user": "User", "assistant": "Trip Buddy"}
    return "\n".join(f"{who.get(m['role'], m['role'])}: {text_of(m['content'])}" for m in messages)


def extract_memories(messages: list[dict], ltm: LongTermMemory, llm) -> list[MemoryItem]:
    """Steps 1 + 2: EXTRACT and CLASSIFY. The LLM returns a filled MemoryList form."""
    chain = EXTRACT_PROMPT | llm.with_structured_output(MemoryList)
    result = chain.invoke({"existing": ltm.as_markdown(), "transcript": transcript_of(messages)})
    return result.memories


def store_memories(items: list[MemoryItem], ltm: LongTermMemory) -> list[dict]:
    """Steps 3 + 4: CONSOLIDATE (ADD / UPDATE / SKIP) and STORE each item in the right drawer."""
    report = []
    for m in items:
        if m.type == "semantic":
            report.append(ltm.set_fact(m.key, m.content))
        elif m.type == "episodic":
            report.append(ltm.add_episode(m.content))
        else:
            report.append(ltm.add_rule(m.content))
    return report


def save_session(messages: list[dict], ltm: LongTermMemory, llm) -> tuple[list[MemoryItem], list[dict]]:
    """The whole pipeline, run when a chat ends: short-term messages → long-term memory."""
    if not messages:
        return [], []
    items = extract_memories(messages, ltm, llm)
    return items, store_memories(items, ltm)
