"""
memory_demo_1.py  -  three kinds of memory feeding one answer   (by Shivank Agarwal)
=====================================================================================
WHAT YOU LEARN
    A customer-service agent built as a LangGraph pipeline of four steps:
      1. semantic memory   -> general facts (return policy, shipping time)
      2. episodic memory   -> what we know about THIS customer (name, preferences, past chats)
      3. procedural memory -> a fixed procedure to look up the customer's orders
      4. generate_response -> the model writes the reply using all three
    The memories here are Python dictionaries so the flow is easy to see.

RUN:  python memory_demo_1.py      (needs OPENAI_API_KEY in .env)
"""
import json
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

# The model that writes the final reply
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")

class AgentState(TypedDict):
    user_id: str
    user_query: str
    order_id: str
    semantic_facts: Dict[str, Any]
    user_context: Dict[str, Any]
    order_details: Dict[str, Any]
    final_response: str

class MemoryDemo:
    def __init__(self):
        # Semantic Memory (general world knowledge/facts)
        self.semantic_memory = {
            "company_policies": {
                "return_period": "30 days",
                "shipping_time": "3-5 business days",
                "customer_service_hours": "9 AM - 6 PM EST"
            },
            "product_categories": {
                "electronics": "Includes phones, laptops, accessories",
                "clothing": "Includes shirts, pants, dresses",
                "books": "Includes fiction, non-fiction, educational"
            }
        }
        
        # Episodic Memory (user-specific experiences/details)
        self.episodic_memory = {
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
            "Frequently buys clothing items",
            "Requested premium shipping upgrade"
        ],
        "preferences": {
            "language": "English", 
            "detail_level": "brief",
            "shipping_speed": "express"
        }
    },
    "user_789": {
        "name": "Carlos Rodriguez",
        "preferred_contact": "phone",
        "recent_interactions": [
            "First-time customer",
            "Purchased books and electronics",
            "Asked about international shipping"
        ],
        "preferences": {
            "language": "Spanish",
            "detail_level": "normal",
            "currency": "EUR"
        }
    }
}
        
        # Simulated data stores (would be external in real scenario)
        self.order_store = {
            "ORD-001": {"status": "shipped", "value": 299.99},
            "ORD-002": {"status": "processing", "value": 149.50},
            "ORD-003": {"status": "delivered", "value": 89.99},
            "ORD-004": {"status": "cancelled", "value": 49.99}
        }
        
        self.user_orders = {
            "user_123": ["ORD-001", "ORD-002"],
            "user_456": ["ORD-003"],
            "user_789": ["ORD-004", "ORD-006"]
        }

    def get_semantic_facts(self, state: AgentState) -> AgentState:
        """Retrieve general world knowledge from semantic memory"""
        print("🔍 Accessing SEMANTIC MEMORY...")
        
        # Extract relevant semantic facts based on query
        relevant_facts = {}
        user_query = state.get("user_query", "").lower()
        
        if "return" in user_query:
            relevant_facts["return_policy"] = self.semantic_memory["company_policies"]["return_period"]
        if "shipping" in user_query:
            relevant_facts["shipping_info"] = self.semantic_memory["company_policies"]["shipping_time"]
        if "electronics" in user_query:
            relevant_facts["category_info"] = self.semantic_memory["product_categories"]["electronics"]
            
        state["semantic_facts"] = relevant_facts
        return state

    def get_user_context(self, state: AgentState) -> AgentState:
        """Retrieve user-specific details from episodic memory"""
        print("👤 Accessing EPISODIC MEMORY...")
        
        user_id = state.get("user_id", "user_123")  # Default for demo
        user_context = self.episodic_memory.get(user_id, {})
        
        state["user_context"] = user_context
        return state

    def procedural_lookup(self, state: AgentState) -> AgentState:
        """Execute procedural memory - follow steps to query data stores"""
        print("🔄 Executing PROCEDURAL MEMORY...")
        
        user_id = state.get("user_id", "user_123")
        order_id = state.get("order_id")
        
        # Procedural steps:
        # 1. Look up user's orders
        user_order_ids = self.user_orders.get(user_id, [])
        
        # 2. If specific order requested, get its details
        if order_id and order_id in user_order_ids:
            order_details = self.order_store.get(order_id, {})
            state["order_details"] = {order_id: order_details}
        # 3. Otherwise get all user orders
        else:
            all_orders = {}
            for oid in user_order_ids:
                all_orders[oid] = self.order_store.get(oid, {})
            state["order_details"] = all_orders
            
        return state

    def generate_response(self, state: AgentState) -> AgentState:
        """Generate final response using all memory types"""
        print("💭 Generating response using all memory types...")
        
        prompt = f"""
        Based on the following information, provide a helpful response to the user:
        
        USER QUERY: {state['user_query']}
        
        SEMANTIC KNOWLEDGE (General Facts):
        {json.dumps(state.get('semantic_facts', {}), indent=2)}
        
        EPISODIC MEMORY (User Context):
        Name: {state['user_context'].get('name', 'Customer')}
        Preferences: {state['user_context'].get('preferences', {})}
        Recent Interactions: {state['user_context'].get('recent_interactions', [])}
        
        PROCEDURAL RESULTS (Order Data):
        {json.dumps(state.get('order_details', {}), indent=2)}
        
        Provide a natural, helpful response that incorporates this information.
        Sign off with Best regards, Agent Nancy.
        """
        
        response = llm.invoke(prompt)
        state["final_response"] = response.text
        return state

    def build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("semantic_memory", self.get_semantic_facts)
        workflow.add_node("episodic_memory", self.get_user_context)
        workflow.add_node("procedural_memory", self.procedural_lookup)
        workflow.add_node("generate_response", self.generate_response)
        
        # Define workflow
        workflow.set_entry_point("semantic_memory")
        workflow.add_edge("semantic_memory", "episodic_memory")
        workflow.add_edge("episodic_memory", "procedural_memory")
        workflow.add_edge("procedural_memory", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()

# Demo execution
def run_demo():
    demo = MemoryDemo()
    workflow = demo.build_workflow()
    
    # Test scenarios
    test_cases = [
        {
            "user_id": "user_123",
            "user_query": "What's the status of my orders and what's your return policy?",
            "order_id": None,
            "semantic_facts": {},
            "user_context": {},
            "order_details": {},
            "final_response": ""
        },
        {
            "user_id": "user_456", 
            "user_query": "Can you check the status of order ORD-003? I might need to return it.",
            "order_id": "ORD-003",
            "semantic_facts": {},
            "user_context": {},
            "order_details": {},
            "final_response": ""
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i}")
        print(f"{'='*50}")
        
        print(f"Query: {test_case['user_query']}")
        print(f"User: {test_case['user_id']}")
        
        result = workflow.invoke(test_case)
        
        print(f"\n📝 FINAL RESPONSE:")
        print(f"{result['final_response']}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    run_demo()