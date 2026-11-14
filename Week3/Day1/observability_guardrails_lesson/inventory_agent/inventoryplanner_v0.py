"""
Inventory Planner  v0  -  one prompt, one answer          (by Shivank Agarwal)
================================================================================
NEW IN THIS VERSION: the simplest possible AI app.
    We type the numbers ourselves, put them in a prompt, and ask the model
    "should I reorder?". No UI, no database, no tools.

RUN:  python inventoryplanner_v0.py
"""
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()   # reads OPENAI_API_KEY from the .env file

# The model: gpt-6-luna with reasoning off = fast, cheap, answers straight away
llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")

# A prompt with {blanks} that we fill in below
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an inventory management expert. Analyze inventory situations "
               "and give clear reorder recommendations."),
    ("human", """Should I reorder {product}? If so, how much?

Current stock: {current_stock} units
Average demand: {average_demand} units per day
Lead time: {lead_time} days

Give your recommendation and reasoning."""),
])

chain = prompt | llm    # "pipe": fill the prompt, then send it to the model

response = chain.invoke({
    "product": "toothpaste",
    "current_stock": 20,
    "average_demand": 5,
    "lead_time": 4,
})
print(response.content)
