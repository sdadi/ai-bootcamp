"""
LangGraph router: send each inventory question to the right specialist.
By Shivank Agarwal. Model: OpenAI gpt-6-luna.

    router ──► reorder  ──► END
           ├─► forecast ──► END
           └─► supplier ──► END

The router is one component of the agent, so it can be evaluated on its own:
did each question reach the right specialist?
"""
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

load_dotenv()                                        # reads OPENAI_API_KEY from .env

llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")


def ask(system, user):
    """One model call with a system prompt; returns the reply as plain text."""
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).text


# ---------- 1. The router: the model must pick exactly one of three labels ----------
class Route(BaseModel):
    category: Literal["reorder", "forecast", "supplier"]


router_llm = llm.with_structured_output(Route)       # the reply is parsed into a Route object


class State(TypedDict):
    query: str
    category: str
    response: str


def router(state):
    decision = router_llm.invoke([
        SystemMessage(content="Route the inventory query to one of: reorder, forecast, or supplier.\n"
                              "- reorder: reorder point, EOQ, safety stock, when to order\n"
                              "- forecast: demand prediction, adjusting forecasts\n"
                              "- supplier: comparing suppliers, lead times, vendor costs"),
        HumanMessage(content=state["query"]),
    ])
    return {"category": decision.category}


# ---------- 2. Three specialists: same model, different system prompts ----------
def reorder_node(state):
    answer = ask("You are an inventory reorder specialist. Answer concisely using inventory "
                 "principles like ROP, EOQ, and safety stock.", state["query"])
    return {"response": answer}


def forecast_node(state):
    answer = ask("You are a demand forecasting specialist. Answer concisely using concepts like "
                 "rolling averages, seasonality, and trend.", state["query"])
    return {"response": answer}


def supplier_node(state):
    answer = ask("You are a supplier management specialist. Answer concisely using concepts like "
                 "lead time, total cost, and vendor reliability.", state["query"])
    return {"response": answer}


def route_edge(state):
    return state["category"]                         # the label decides which node runs next


# ---------- 3. Wire the graph ----------
graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("reorder", reorder_node)
graph.add_node("forecast", forecast_node)
graph.add_node("supplier", supplier_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", route_edge,
                            {"reorder": "reorder", "forecast": "forecast", "supplier": "supplier"})
graph.add_edge("reorder", END)
graph.add_edge("forecast", END)
graph.add_edge("supplier", END)

app = graph.compile()

queries = [
    "We have 300 units on hand and sell 30 a day with a 5-day lead time. Should we reorder?",
    "Demand spiked 60% last week. How should I adjust my forecast?",
    "Supplier A is cheaper but takes 30 days. Supplier B costs more but delivers in 5 days. Which is better?",
]

if __name__ == "__main__":
    for i, q in enumerate(queries, start=1):
        result = app.invoke({"query": q, "category": "", "response": ""})
        print(f"\n[{i}] {q}")
        print(f"    route  → {result['category']}")
        answer = result["response"].strip().replace("\n", "\n" + " " * 13)   # keep long answers indented
        print(f"    answer → {answer}")
        print("-" * 70)
