"""
Inventory Planner  v2  -  the agent reads a real database (RAG + tools)   (by Shivank Agarwal)
================================================================================
NEW IN THIS VERSION (v0 prompt -> v1 web form -> v2 database + tools + agent):
    The user no longer types the numbers. An AI AGENT gets them itself:
    it calls TOOLS (Python functions) that read a SQLite database, then answers.
    This is Retrieval-Augmented Generation (RAG): fetch real data, then generate.
    The UI becomes a chat.

RUN:  python inventoryplanner_v2.py   -> open the link it prints (http://127.0.0.1:7860)
"""

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from database import init_database, get_db_connection

load_dotenv()   # reads OPENAI_API_KEY (and the other settings) from the .env file

if not init_database():
    raise RuntimeError("Could not create the inventory database")

# reasoning_effort="none" -> fast, cheap answers with no hidden "thinking" tokens
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")


# ------------------------------------------------------------------
# Tools: Python functions the AI agent may call to read the database
# (the docstring is what the model reads to decide WHEN to use each tool)
# ------------------------------------------------------------------
@tool
def get_inventory_data(product_name: str) -> str:
    """Get stock, average daily demand and lead time for ONE product from the database."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT product_name, current_stock, average_demand, lead_time "
        "FROM inventory WHERE product_name LIKE ?", (f"%{product_name}%",)
    ).fetchone()
    conn.close()
    if not row:
        return f"Product '{product_name}' not found in inventory database"
    return (f"Product: {row[0]}, Current Stock: {row[1]} units, "
            f"Average Demand: {row[2]} units/day, Lead Time: {row[3]} days")


@tool
def get_all_inventory() -> str:
    """Get the status of ALL products, marking the ones that are running low."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT product_name, current_stock, average_demand, lead_time "
        "FROM inventory ORDER BY product_name"
    ).fetchall()
    conn.close()
    if not rows:
        return "No inventory data found"
    lines = []
    for name, stock, demand, lead_time in rows:
        days_remaining = stock / demand if demand > 0 else 0
        status = "LOW STOCK" if days_remaining <= lead_time else "OK"
        lines.append(f"{name}: {stock} units, {days_remaining:.1f} days remaining ({status})")
    return "Current inventory status:\n" + "\n".join(lines)


# ------------------------------------------------------------------
# The agent: the model + the tools + the instructions
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are an inventory management expert with access to a real-time inventory database.

When a user asks about an inventory decision:
1. First get the current data for the product with get_inventory_data
2. Decide if reordering is needed
3. Give a clear recommendation with reasoning

Key rules:
- Days remaining = current_stock / average_demand
- If days remaining <= lead time, recommend reordering
- Recommend ordering at least average_demand x lead_time units
- For general questions, use get_all_inventory to show the overall status

Always base your answer on the database data, not on assumptions. Be clear and concise."""

tools = [get_inventory_data, get_all_inventory]
agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def run_agent(question: str) -> str:
    """Ask the agent one question. Prints each tool call so you can follow along in the terminal."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"[tool call] {call['name']}({call['args']})")
    return result["messages"][-1].text     # the agent's final answer, as plain text


def inventory_chat(message: str) -> str:
    """Everything that happens for one user question."""
    if not message.strip():
        return "Please ask a question about inventory."

    try:
        answer = run_agent(message)                        # the agent + tools + database
    except Exception as e:
        return f"Sorry, I encountered an error: {e}"
    return answer


# ------------------------------------------------------------------
# The web page (Gradio)
# ------------------------------------------------------------------
with gr.Blocks(title="Inventory Management Assistant") as demo:
    gr.Markdown("# 📦 Inventory Management Assistant")
    gr.Markdown("Ask in plain English. The agent reads the live inventory database to answer.")

    chatbot = gr.Chatbot(label="Inventory Assistant", height=400)
    msg = gr.Textbox(label="Ask about inventory", placeholder="e.g. Should I reorder toothpaste?", lines=2)
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear chat")

    gr.Examples(examples=['Should I reorder toothpaste?', 'Check the inventory status for shampoo', 'Show me all inventory with low stock'], inputs=msg, label="Example questions")

    def respond(message, chat_history):
        if not message.strip():
            return chat_history, ""
        answer = inventory_chat(message)
        chat_history = chat_history + [{"role": "user", "content": message},
                                       {"role": "assistant", "content": answer}]
        return chat_history, ""

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
    clear_btn.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
