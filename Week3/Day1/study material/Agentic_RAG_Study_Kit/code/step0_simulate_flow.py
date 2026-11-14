"""
step0_simulate_flow.py  -  Feel the Agentic RAG flow BEFORE using real AI
========================================================================
Created by Shivank Agrawal

No API keys. No installs. Pure Python. Just run:
    python step0_simulate_flow.py

Here the "agents" are simple Python functions and the "embeddings" are
simple word counts, so you can SEE every step clearly:

  question -> Router -> Retriever (RAG tool or Web tool) -> Answer Grader
           -> Hallucination Grader -> Answer Generator -> final answer

Once this makes sense, main.py does the SAME thing with real LLMs.
"""

import math
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Our tiny "private PDF" (in real life this is a long document)
# ---------------------------------------------------------------------------
DOCUMENT = """
The Transformer is a neural network architecture based entirely on attention.
It removes recurrence and convolutions completely, so it trains much faster.

Self-attention lets each word in a sentence look at every other word and decide
how much attention to pay to it. This is how the model understands context.

Attention uses three vectors for every word: a query, a key and a value.
The score between two words is the dot product of the query and the key,
scaled and passed through a softmax.

Multi-head attention runs several attention operations in parallel so the model
can focus on different kinds of relationships at the same time.

Transformers have no recurrence, so positional encodings are added to the
input embeddings to tell the model the order of the words.
"""

# Fake "internet" for the web search tool
FAKE_WEB = {
    "weather": "Web result: Today in Bengaluru it is 24 degrees C with light rain.",
    "cricket": "Web result: India won yesterday's match by 6 wickets.",
    "multi-head": "Web result: Multi-head attention splits attention into several "
                  "'heads' that each learn a different relationship, then joins them.",
}

STOP_WORDS = set("the a an is of to and in it its for on what how does do are with by "
                 "that this be at as from can each every".split())


def words(text: str) -> list[str]:
    """Lower-case, drop common words, and cut a plural 's' (transformers -> transformer)."""
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in STOP_WORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


# ---------------------------------------------------------------------------
# INDEXING: split -> embed -> store   (done once)
# ---------------------------------------------------------------------------
def split_into_chunks(text: str) -> list[str]:
    """Simplest chunking strategy: one paragraph = one chunk."""
    paragraphs = text.strip().split("\n\n")
    return [" ".join(p.split()) for p in paragraphs if p.strip()]


def embed(text: str) -> Counter:
    """Toy embedding: count of each word. Real embeddings are 384+ numbers."""
    return Counter(words(text))


def cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[w] * b[w] for w in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


CHUNKS = split_into_chunks(DOCUMENT)
VECTOR_DB = [(chunk, embed(chunk)) for chunk in CHUNKS]


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------
def rag_tool(question: str, top_k: int = 2) -> list[tuple[float, str]]:
    q = embed(question)
    scored = sorted(((cosine(q, vec), chunk) for chunk, vec in VECTOR_DB), reverse=True)
    return [(round(s, 2), c) for s, c in scored[:top_k] if s > 0]


def web_search_tool(question: str) -> str:
    for keyword, result in FAKE_WEB.items():
        if keyword in question.lower():
            return result
    return "Web result: (nothing found in our fake internet)"


# ---------------------------------------------------------------------------
# AGENTS (simple rules here; real agents use an LLM to decide)
# ---------------------------------------------------------------------------
def router_agent(question: str) -> str:
    best = max(cosine(embed(question), vec) for _, vec in VECTOR_DB)
    return "vectorstore" if best >= 0.15 else "websearch"


def retriever_agent(question: str, route: str) -> dict:
    if route == "vectorstore":
        hits = rag_tool(question)
        if hits:
            return {"source": "document", "answer": hits[0][1], "evidence": hits}
    return {"source": "web", "answer": web_search_tool(question), "evidence": []}


def answer_grader_agent(question: str, retrieved: dict) -> str:
    overlap = set(words(question)) & set(words(retrieved["answer"]))
    return "yes" if overlap or retrieved["source"] == "web" else "no"


def hallucination_grader_agent(retrieved: dict) -> str:
    if retrieved["source"] == "web":
        return "yes"
    supporting_text = " ".join(c for _, c in retrieved["evidence"])
    return "yes" if retrieved["answer"] in supporting_text else "no"


def answer_generator_agent(question: str, retrieved: dict, relevant: str, grounded: str) -> str:
    if relevant == "yes" and grounded == "yes":
        return retrieved["answer"]
    return web_search_tool(question)


# ---------------------------------------------------------------------------
# THE CREW (sequential process)
# ---------------------------------------------------------------------------
def run(question: str, make_it_lie: bool = False) -> None:
    print("\n" + "=" * 72)
    print("QUESTION:", question)
    route = router_agent(question)
    print(f"  1. Router            -> {route}")
    retrieved = retriever_agent(question, route)
    if make_it_lie:
        # Pretend the LLM "added" a fact that is NOT in the document
        retrieved["answer"] += " It was invented in 1975 by a team in Mumbai."
        print("     (demo: the retriever added a made-up fact!)")
    print(f"  2. Retriever         -> source={retrieved['source']}")
    for score, chunk in retrieved["evidence"]:
        print(f"        chunk (similarity {score}): {chunk[:70]}...")
    relevant = answer_grader_agent(question, retrieved)
    print(f"  3. Answer Grader     -> relevant? {relevant}")
    grounded = hallucination_grader_agent(retrieved)
    print(f"  4. Hallucination Grd -> grounded? {grounded}")
    final = answer_generator_agent(question, retrieved, relevant, grounded)
    print(f"  5. Answer Generator  -> {final}")


if __name__ == "__main__":
    print(f"Indexed {len(CHUNKS)} chunks into our toy vector database.")
    run("How does self-attention work in a Transformer?")
    run("Why do Transformers need positional encoding?")
    run("What is the weather today in Bengaluru?")
    run("Who won the cricket match yesterday?")
    # Watch the Hallucination Grader catch a made-up fact:
    run("What is multi-head attention?", make_it_lie=True)
