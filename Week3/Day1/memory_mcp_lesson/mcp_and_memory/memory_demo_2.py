"""
memory_demo_2.py  -  memory that updates itself and is saved to disk   (by Shivank Agarwal)
===========================================================================================
WHAT YOU LEARN
    Same four-step agent as memory_demo_1.py, plus a new step: update_memory.
    In every turn the model checks whether the user changed a preference
    ("keep it brief from now on"). If so, the preference is updated straight away
    (the "hot path"), an audit trail is kept, and everything is saved to
    episodic_memory.json so it survives after the program ends.

RUN:  python memory_demo_2.py      (needs OPENAI_API_KEY in .env)
"""
import copy
import json
import os
from typing import Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")
# Same model, but forced to reply with a valid JSON object (used by the preference detector)
json_llm = llm.bind(response_format={"type": "json_object"})

class AgentState(TypedDict):
    user_id: str
    user_query: str
    order_id: Optional[str]
    semantic_facts: Dict[str, Any]
    user_context: Dict[str, Any]
    order_details: Dict[str, Any]
    preference_update: Dict[str, Any]   # newly detected preference changes
    final_response: str


EPISODIC_MEMORY_FILE = os.path.join(os.path.dirname(__file__), "episodic_memory.json")

EPISODIC_MEMORY_DEFAULT = {
    "user_123": {
        "name": "Alice Johnson",
        "preferred_contact": "email",
        "recent_interactions": [
            "Asked about return policy yesterday",
            "Purchased electronics last week"
        ],
        "preferences": {
            "language": "English",
            "detail_level": "detailed"
        }
    },
    "user_456": {
        "name": "Bob Smith",
        "preferred_contact": "sms",
        "recent_interactions": [
            "Complained about delayed shipping last month",
            "Frequently buys clothing items"
        ],
        "preferences": {
            "language": "English",
            "detail_level": "brief",
            "shipping_speed": "express"
        }
    }
}


class MemoryDemo:
    def __init__(self):
        # Reset to default and write to JSON file so we start fresh each demo
        self.episodic_memory: Dict[str, Any] = copy.deepcopy(EPISODIC_MEMORY_DEFAULT)   # a fresh copy, not the original
        with open(EPISODIC_MEMORY_FILE, "w") as f:
            json.dump(self.episodic_memory, f, indent=2)
        print(f"✅ Episodic memory written to {EPISODIC_MEMORY_FILE}")
        self.turn = 0

        self.semantic_memory = {
            "company_policies": {
                "return_period": "30 days",
                "shipping_time": "3-5 business days",
                "customer_service_hours": "9 AM - 6 PM EST"
            },
            "product_categories": {
                "electronics": "Includes phones, laptops, accessories",
                "clothing": "Includes shirts, pants, dresses"
            }
        }

        self.order_store = {
            "ORD-001": {"status": "shipped",    "value": 299.99},
            "ORD-002": {"status": "processing", "value": 149.50},
            "ORD-003": {"status": "delivered",  "value": 89.99},
        }
        self.user_orders = {
            "user_123": ["ORD-001", "ORD-002"],
            "user_456": ["ORD-003"],
        }

    # ── Nodes ────────────────────────────────────────────────────────────────

    def get_semantic_facts(self, state: AgentState) -> AgentState:
        print("🔍 Accessing SEMANTIC MEMORY...")
        relevant_facts = {}
        query = state.get("user_query", "").lower()
        if "return"     in query: relevant_facts["return_policy"]  = self.semantic_memory["company_policies"]["return_period"]
        if "shipping"   in query: relevant_facts["shipping_info"]  = self.semantic_memory["company_policies"]["shipping_time"]
        if "electronics" in query: relevant_facts["category_info"] = self.semantic_memory["product_categories"]["electronics"]
        state["semantic_facts"] = relevant_facts
        return state

    def get_user_context(self, state: AgentState) -> AgentState:
        print("👤 Accessing EPISODIC MEMORY...")
        state["user_context"] = self.episodic_memory.get(state["user_id"], {})
        return state

    def detect_and_update_preference(self, state: AgentState) -> AgentState:
        """
        Use the LLM to detect preference change requests in the user query.
        If a preference change is found, update episodic memory immediately
        so the SAME turn already reflects the new preference.
        """
        print("🔄 Checking for PREFERENCE UPDATES...")
        self.turn += 1

        prompt = f"""
You are a preference-change detector. Analyse the user message below.
If the user is requesting a change to any of their preferences (e.g. detail level,
language, contact method, shipping speed, etc.), respond with ONLY a valid JSON object
mapping preference key → new value, like:
{{"detail_level": "brief"}}

If there is NO preference change, respond with ONLY an empty JSON object: {{}}

User message: "{state['user_query']}"
"""
        result = json_llm.invoke(prompt)
        try:
            updates: Dict[str, Any] = json.loads(result.text)
        except json.JSONDecodeError:
            updates = {}
        if not isinstance(updates, dict):
            updates = {}

        state["preference_update"] = updates

        if updates:
            user_id = state["user_id"]
            if user_id in self.episodic_memory:
                old_prefs = {k: self.episodic_memory[user_id]["preferences"][k]
                             for k in updates if k in self.episodic_memory[user_id]["preferences"]}
                self.episodic_memory[user_id]["preferences"].update(updates)
                # Append to audit trail
                audit = self.episodic_memory[user_id].setdefault("preference_history", {})
                if not audit:
                    audit[f"turn_{self.turn - 1}"] = old_prefs  # seed state before first change
                audit[f"turn_{self.turn}"] = dict(updates)
                # Refresh the user_context so generate_response uses new prefs
                state["user_context"] = self.episodic_memory[user_id]
                print(f"   ✏️  Preference updated for {user_id}: {old_prefs} → {updates}")
                print(f"   📋 New preferences: {self.episodic_memory[user_id]['preferences']}")
                # Persist updated memory to JSON file
                with open(EPISODIC_MEMORY_FILE, "w") as f:
                    json.dump(self.episodic_memory, f, indent=2)
                print(f"   💾 Episodic memory saved to {EPISODIC_MEMORY_FILE}")
        else:
            print("   — No preference change detected.")

        return state

    def procedural_lookup(self, state: AgentState) -> AgentState:
        print("📦 Executing PROCEDURAL MEMORY...")
        user_id  = state["user_id"]
        order_id = state.get("order_id")
        user_order_ids = self.user_orders.get(user_id, [])

        if order_id and order_id in user_order_ids:
            state["order_details"] = {order_id: self.order_store[order_id]}
        else:
            state["order_details"] = {oid: self.order_store[oid] for oid in user_order_ids if oid in self.order_store}
        return state

    def generate_response(self, state: AgentState) -> AgentState:
        print("💭 Generating response using all memory types...")
        prefs       = state["user_context"].get("preferences", {})
        detail_level = prefs.get("detail_level", "normal")

        prompt = f"""
You are a helpful customer service agent.
Respond to the user query below.  Detail level: {detail_level}.
{"Keep your answer SHORT and to the point." if detail_level == "brief" else "Provide a thorough, detailed answer." if detail_level == "detailed" else ""}

USER QUERY: {state['user_query']}

SEMANTIC KNOWLEDGE:
{json.dumps(state.get('semantic_facts', {}), indent=2)}

EPISODIC MEMORY (User Context):
Name: {state['user_context'].get('name', 'Customer')}
Preferences: {json.dumps(prefs)}
Recent Interactions: {state['user_context'].get('recent_interactions', [])}

PROCEDURAL RESULTS (Order Data):
{json.dumps(state.get('order_details', {}), indent=2)}

Sign off with: Best regards, Agent Nancy.
"""
        response = llm.invoke(prompt)
        state["final_response"] = response.text
        return state

    # ── Graph ────────────────────────────────────────────────────────────────

    def build_workflow(self):
        wf = StateGraph(AgentState)

        wf.add_node("semantic_memory",  self.get_semantic_facts)
        wf.add_node("episodic_memory",  self.get_user_context)
        wf.add_node("update_memory",    self.detect_and_update_preference)   
        wf.add_node("procedural_memory", self.procedural_lookup)
        wf.add_node("generate_response", self.generate_response)

        wf.set_entry_point("semantic_memory")
        wf.add_edge("semantic_memory",   "episodic_memory")
        wf.add_edge("episodic_memory",   "update_memory")        # ← new edge
        wf.add_edge("update_memory",     "procedural_memory")
        wf.add_edge("procedural_memory", "generate_response")
        wf.add_edge("generate_response", END)

        return wf.compile()


# ── Demo ─────────────────────────────────────────────────────────────────────

def make_state(user_id: str, query: str, order_id: Optional[str] = None) -> AgentState:
    return {
        "user_id":          user_id,
        "user_query":       query,
        "order_id":         order_id,
        "semantic_facts":   {},
        "user_context":     {},
        "order_details":    {},
        "preference_update": {},
        "final_response":   ""
    }


def run_demo():
    demo     = MemoryDemo()
    workflow = demo.build_workflow()

    print("\n" + "="*60)
    print("TURN 1  — Normal query (detail_level = 'detailed')")
    print("="*60)
    result1 = workflow.invoke(make_state(
        user_id="user_123",
        query="What is the status of my orders and what's the return policy?"
    ))
    print(f"\n📝 RESPONSE:\n{result1['final_response']}\n")

    print("="*60)
    print("TURN 2  — User requests preference change to 'brief'")
    print("="*60)
    result2 = workflow.invoke(make_state(
        user_id="user_123",
        query="Please keep your answers brief from now on. Also, what's the shipping time?"
    ))
    print(f"\n📝 RESPONSE:\n{result2['final_response']}\n")
    print(f"🗂️  Stored preference after Turn 2: {demo.episodic_memory['user_123']['preferences']}")
    print(f"📂 Open '{EPISODIC_MEMORY_FILE}' to see the updated JSON on disk.\n")

    print("="*60)
    print("TURN 3  — Follow-up query, should now respect 'brief' preference")
    print("="*60)
    result3 = workflow.invoke(make_state(
        user_id="user_123",
        query="Can you give me an update on all my orders?"
    ))
    print(f"\n📝 RESPONSE:\n{result3['final_response']}\n")


if __name__ == "__main__":
    run_demo()
