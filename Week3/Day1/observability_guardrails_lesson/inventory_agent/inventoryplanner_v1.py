"""
Inventory Planner  v1  -  a web form around the prompt     (by Shivank Agarwal)
================================================================================
NEW IN THIS VERSION: a Gradio web page.
    Same prompt as v0, but now anyone can type a product and its numbers into a
    form and click a button. The user still has to KNOW the numbers.

RUN:  python inventoryplanner_v1.py   -> open the link it prints (http://127.0.0.1:7860)
"""
import gradio as gr
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an inventory management expert. Analyze inventory situations "
               "and give clear reorder recommendations with reasoning."),
    ("human", """Should I reorder {product}? If so, how much?

Current stock: {current_stock} units
Average demand: {average_demand} units per day
Lead time: {lead_time} days

Give your recommendation and reasoning."""),
])
chain = prompt | llm


def inventory_chat(product, current_stock, average_demand, lead_time):
    """Called when the button is clicked: fill the prompt with the form values."""
    try:
        response = chain.invoke({
            "product": product,
            "current_stock": int(current_stock),
            "average_demand": float(average_demand),
            "lead_time": int(lead_time),
        })
        return response.content
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="Inventory Management Assistant") as demo:
    gr.Markdown("# 📦 Inventory Management Assistant")
    gr.Markdown("Type the numbers for a product and get an AI reorder recommendation.")

    with gr.Row():
        with gr.Column():
            product_input = gr.Textbox(label="Product name", value="toothpaste")
            stock_input = gr.Number(label="Current stock (units)", value=20, minimum=0)
            demand_input = gr.Number(label="Average daily demand (units/day)", value=5.0, minimum=0.1)
            leadtime_input = gr.Number(label="Lead time (days)", value=4, minimum=1)
            analyze_btn = gr.Button("Analyze Inventory", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="AI recommendation", lines=10, interactive=False)

    analyze_btn.click(fn=inventory_chat,
                      inputs=[product_input, stock_input, demand_input, leadtime_input],
                      outputs=output)

    gr.Examples(examples=[["toothpaste", 20, 5.0, 4], ["shampoo", 15, 3.0, 5], ["soap", 50, 8.0, 3]],
                inputs=[product_input, stock_input, demand_input, leadtime_input])

if __name__ == "__main__":
    demo.launch()
