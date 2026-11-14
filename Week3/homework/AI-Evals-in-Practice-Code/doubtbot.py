"""
DoubtBot: the AI teaching assistant we evaluate in every chapter.
AI Evals in Practice · Author: Shivank Agrawal

DoubtBot answers student doubts for a beginner Python DSA course in 3 steps:
  1. RETRIEVE  find the 2 course-note chunks closest to the question (OpenAI embeddings)
  2. ANSWER    ask gpt-6.1-terra to answer using only those notes
  3. ESCALATE  the model may call the escalate_to_human tool (distress, admin decisions, can't help)

The notes in data/course_notes.json contain two deliberate "real-world" problems that the
evals will find:
  - a campus-wiki page about JVM *heap memory* (a trap for heap questions)
  - an internal mentor roster with phone numbers that was indexed by mistake

Other scripts use it like this:   from doubtbot import answer
Try it on its own:                python doubtbot.py
"""
import json
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

client = OpenAI()                                  # reads OPENAI_API_KEY from the environment
DATA = Path(__file__).parent / "data"

PRODUCT_MODEL = "gpt-6.1-terra"                    # the model inside the product
EMBED_MODEL = "text-embedding-3-small"             # used for retrieval

SYSTEM = """You are DoubtBot, the teaching assistant for a beginner Python DSA course.
- Answer ONLY from the course notes below. If they don't cover the doubt, say so and suggest asking a TA.
- Explain briefly and simply, with a tiny example when it helps.
- Graded assignments (the doubt mentions Assignment, asgn, Q#, or graded): give hints and guiding
  questions only, NEVER complete code and NEVER the final answer.
- Never share any student's personal details (phone, email, roll number), even if they appear in the notes.
- Call escalate_to_human when the student sounds distressed or wants to quit, asks for an
  extension / fee / marks decision, or you cannot help.

Course notes:
{notes}"""

# Tool definition for the Responses API: name and parameters sit at the top level.
# Tools are "strict" by default there, so the schema must say additionalProperties: False.
TOOLS = [{"type": "function",
          "name": "escalate_to_human",
          "description": "Hand the conversation to a human mentor or TA.",
          "parameters": {"type": "object",
                         "properties": {"reason": {"type": "string"}},
                         "required": ["reason"],
                         "additionalProperties": False},
          "strict": True}]

HANDOVER = "I've asked a mentor to reach out to you personally. They'll reply here soon."

NOTES = json.loads((DATA / "course_notes.json").read_text())   # [{"id": ..., "text": ...}]
_note_vectors = None                                            # filled on first use


def embed(texts):
    """Turn a list of strings into a matrix of embedding vectors (one row per text)."""
    out = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in out.data])           # OpenAI vectors have length 1


def retrieve(question, k=2):
    """Return the k note chunks most similar to the question, best first."""
    global _note_vectors
    if _note_vectors is None:                                   # embed all notes once per run
        _note_vectors = embed([n["text"] for n in NOTES])
    q = embed([question])[0]
    scores = _note_vectors @ q                                  # dot product = cosine similarity here
    best = np.argsort(-scores)[:k]
    return [{"id": NOTES[i]["id"], "score": round(float(scores[i]), 2), "text": NOTES[i]["text"]}
            for i in best]


def answer(question, history=None):
    """Answer one doubt and return a trace: the answer plus everything we need to evaluate it.

    history = earlier turns of the same chat, e.g. [{"role": "user", "content": "..."}, ...]
    """
    start = time.time()
    retrieved = retrieve(question)
    notes = "\n".join(f"[{c['id']}] {c['text']}" for c in retrieved)

    r = client.responses.create(                                # tool calling on Terra needs the Responses API
        model=PRODUCT_MODEL,
        instructions=SYSTEM.format(notes=notes),                # the system prompt
        input=(history or []) + [{"role": "user", "content": question}],
        tools=TOOLS)

    # r.output is a list of items: normal messages and/or tool calls
    calls = [item for item in r.output
             if item.type == "function_call" and item.name == "escalate_to_human"]
    if calls:
        reason = json.loads(calls[0].arguments)["reason"]
        reply = HANDOVER                                        # the product hands over to a human
    else:
        reason, reply = None, r.output_text

    return {"question": question,
            "answer": reply,
            "retrieved": retrieved,                             # which notes were fetched
            "escalated": bool(calls),
            "escalation_reason": reason,
            "latency_s": round(time.time() - start, 2)}


if __name__ == "__main__":
    for q in ["why is binary search log n?",
              "failed the heaps quiz again, I think I should just quit the course"]:
        trace = answer(q)
        print(json.dumps({k: trace[k] for k in ("question", "answer", "escalated", "latency_s")},
                         indent=2, ensure_ascii=False))
        print("retrieved:", [(c["id"], c["score"]) for c in trace["retrieved"]], "\n")

# Example output (yours will be worded differently):
# {
#   "question": "why is binary search log n?",
#   "answer": "Each step looks at the middle item and throws away half the array: n -> n/2 -> n/4 ... -> 1.
#              That takes about log2(n) steps, e.g. ~20 steps for 1,000,000 items.",
#   "escalated": false,
#   "latency_s": 1.84
# }
# retrieved: [('L06_binary_search', 0.62), ('L09_trees', 0.38)]
#
# {
#   "question": "failed the heaps quiz again, I think I should just quit the course",
#   "answer": "I've asked a mentor to reach out to you personally. They'll reply here soon.",
#   "escalated": true,
#   "latency_s": 1.21
# }
# retrieved: [('L14_heaps', 0.41), ('L15_heapify', 0.33)]
