"""
Observability demo with LangSmith  -  by Shivank Agarwal
=========================================================

WHAT IT DOES
    A tiny "personality bot": you give it a famous person's name and it answers in 2 lines.
    Two caches make repeated questions free, and LangSmith records every step, so for
    each question you can open a TRACE and see exactly what happened, how long it took
    and what it cost.

        question
           |
           v
    [check_exact_cache] ----- hit ----> END   (same question asked before: free, instant)
           | miss
           v
    [check_semantic_cache] -- hit ----> END   (a question with the same MEANING: almost free)
           | miss
           v
    [call_llm] -----------------------> END   (ask OpenAI, then save the answer in both caches)

THE 4 MELT PILLARS, AND WHERE THEY ARE IN THIS FILE
    Metrics -> tokens, cost and time for every question, plus session totals at the end
    Events  -> record_event("cache_hit_exact", ...) etc. appear as steps in the trace
    Logs    -> wrap_openai() stores every prompt and every answer inside the trace
    Traces  -> @traceable + LangGraph: one tree per question in LangSmith

HOW TO RUN
    1.  pip install -r requirements.txt
    2.  Create a free LangSmith account at https://smith.langchain.com -> Settings -> API Keys
    3.  Copy .env.example to .env and fill in your two keys
    4.  python obs_demo_langsmith.py
    5.  Open https://smith.langchain.com -> Tracing Projects -> observability_demo
"""
import time
from typing import TypedDict

import numpy as np
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

load_dotenv()   # reads OPENAI_API_KEY and the LANGSMITH_* settings from .env

CHAT_MODEL = "gpt-6-luna"                    # OpenAI's small, fast, cheap model
EMBEDDING_MODEL = "text-embedding-3-small"   # turns text into a list of numbers (a vector)

# Prices in US$ per 1 million tokens (check OpenAI's pricing page, they change)
PRICE_INPUT, PRICE_OUTPUT, PRICE_EMBEDDING = 0.10, 0.50, 0.02

SIMILARITY_THRESHOLD = 0.70   # how close (0 to 1) a new question must be to reuse an old answer

# wrap_openai: one line, and every OpenAI call (prompt, answer, model, tokens, time)
# is logged to LangSmith automatically.
client = wrap_openai(OpenAI())


# ---------------------------------------------------------------------------
# The State: what the graph knows about ONE question
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    query: str            # the question, e.g. "Elon Musk"
    response: str         # the answer
    cache_hit_type: str   # "exact", "semantic", "fresh" (or "miss" while still looking)
    similarity: float     # best similarity score found in the semantic cache
    cost: float           # US$ spent on the LLM for this question
    tokens: int           # LLM tokens used for this question


def clean(text: str) -> str:
    """'  Marie Curie ' and 'marie curie' should count as the same question."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# The Agent: caches + the OpenAI calls, each one traced
# ---------------------------------------------------------------------------
class PersonalityAgent:
    def __init__(self):
        self.cache = {}        # cleaned question -> answer
        self.embeddings = {}   # cleaned question -> its embedding vector

    @traceable(name="get_embedding", run_type="embedding")
    def get_embedding(self, text: str) -> np.ndarray:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        tokens = response.usage.total_tokens
        print(f"   [metric] embedding: {tokens} tokens, ${tokens / 1e6 * PRICE_EMBEDDING:.7f}")
        return np.array(response.data[0].embedding)

    @traceable(name="find_semantic_match", run_type="tool")
    def find_semantic_match(self, query: str):
        """Return (closest saved question, similarity score)."""
        if not self.embeddings:
            return None, 0.0
        query_vector = self.get_embedding(clean(query))
        best_key, best_score = None, -1.0          # start below any real score
        for key, vector in self.embeddings.items():
            score = float(np.dot(query_vector, vector))   # OpenAI vectors have length 1, so dot = cosine similarity
            if score > best_score:
                best_key, best_score = key, score
        print(f"   [log] closest saved question: '{best_key}' (similarity {best_score:.2f})")
        return best_key, best_score

    @traceable(name="call_openai_model", run_type="chain")
    def call_openai_model(self, question: str):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            reasoning_effort="none",   # answer straight away, no hidden "thinking" tokens
            messages=[
                {"role": "system", "content": "Give exactly 2 short lines about this person's personality and achievements."},
                {"role": "user", "content": f"Tell me about {question}"},
            ],
        )
        usage = response.usage
        cost = usage.prompt_tokens / 1e6 * PRICE_INPUT + usage.completion_tokens / 1e6 * PRICE_OUTPUT
        return response.choices[0].message.content.strip(), usage.total_tokens, cost

    @traceable(name="record_event", run_type="tool")
    def record_event(self, event_name: str, details: dict):
        """An EVENT: a meaningful moment we want to find and count later."""
        print(f"   [event] {event_name} {details}")


# ---------------------------------------------------------------------------
# The graph: 3 nodes + 2 routers  (LangGraph traces every node automatically)
# ---------------------------------------------------------------------------
def build_graph(agent: PersonalityAgent):

    def check_exact_cache(state: AgentState) -> dict:
        key = clean(state["query"])
        if key in agent.cache:
            agent.record_event("cache_hit_exact", {"question": key})
            return {"response": agent.cache[key], "cache_hit_type": "exact", "cost": 0.0, "tokens": 0}
        return {"cache_hit_type": "miss"}

    def check_semantic_cache(state: AgentState) -> dict:
        best_key, score = agent.find_semantic_match(state["query"])
        if best_key and score >= SIMILARITY_THRESHOLD:
            agent.record_event("cache_hit_semantic",
                               {"question": state["query"], "matched": best_key, "similarity": round(score, 2)})
            # Only a tiny embedding cost was spent (printed above), no LLM cost
            return {"response": agent.cache[best_key], "cache_hit_type": "semantic",
                    "similarity": score, "cost": 0.0, "tokens": 0}
        return {"cache_hit_type": "miss", "similarity": score}

    def call_llm(state: AgentState) -> dict:
        question = state["query"]
        agent.record_event("llm_call_started", {"question": question})
        start = time.perf_counter()
        answer, tokens, cost = agent.call_openai_model(question)
        seconds = time.perf_counter() - start

        key = clean(question)                       # save for next time, in BOTH caches
        agent.cache[key] = answer
        agent.embeddings[key] = agent.get_embedding(key)
        agent.record_event("answer_cached", {"question": key, "llm_seconds": round(seconds, 2)})
        return {"response": answer, "cache_hit_type": "fresh", "cost": cost, "tokens": tokens}

    def after_exact(state: AgentState) -> str:
        return END if state["cache_hit_type"] == "exact" else "check_semantic_cache"

    def after_semantic(state: AgentState) -> str:
        return END if state["cache_hit_type"] == "semantic" else "call_llm"

    graph = StateGraph(AgentState)
    graph.add_node("check_exact_cache", check_exact_cache)
    graph.add_node("check_semantic_cache", check_semantic_cache)
    graph.add_node("call_llm", call_llm)
    graph.set_entry_point("check_exact_cache")
    graph.add_conditional_edges("check_exact_cache", after_exact)
    graph.add_conditional_edges("check_semantic_cache", after_semantic)
    graph.add_edge("call_llm", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# One question = one trace in LangSmith
# ---------------------------------------------------------------------------
@traceable(name="personality_query", run_type="chain", metadata={"app": "personality_bot", "version": "2.0"})
def run_query(graph, query: str) -> dict:
    start = time.perf_counter()
    result = graph.invoke({"query": query})
    seconds = time.perf_counter() - start
    print(f"   [metric] {result['cache_hit_type'].upper()} | {seconds:.2f}s | "
          f"{result.get('tokens', 0)} tokens | ${result.get('cost', 0.0):.6f}")
    print(f"   -> {result['response']}")
    return result


if __name__ == "__main__":
    agent = PersonalityAgent()
    graph = build_graph(agent)

    questions = [
        "Elon Musk",                   # fresh: asked for the first time
        "Elon Musk",                   # exact hit: same text
        "Who is Elon Musk?",           # semantic hit (if similarity >= threshold): same meaning
        "Marie Curie",                 # fresh
        "  marie curie ",              # exact hit after cleaning (lower case, extra spaces)
        "Tell me about Marie Curie",   # semantic hit (if similarity >= threshold)
        "Albert Einstein",             # fresh
    ]

    counts = {"exact": 0, "semantic": 0, "fresh": 0}
    total_cost, total_tokens = 0.0, 0
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question!r}")
        result = run_query(graph, question)
        counts[result["cache_hit_type"]] += 1
        total_cost += result.get("cost", 0.0)
        total_tokens += result.get("tokens", 0)

    print("\n=== SESSION METRICS ===")
    print(f"Questions        : {len(questions)}")
    print(f"Exact cache hits : {counts['exact']}")
    print(f"Semantic hits    : {counts['semantic']}")
    print(f"LLM calls        : {counts['fresh']}")
    print(f"Cache hit rate   : {100 * (counts['exact'] + counts['semantic']) / len(questions):.0f}%")
    print(f"LLM tokens       : {total_tokens}")
    print(f"LLM cost         : ${total_cost:.6f}")
