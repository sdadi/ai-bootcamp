"""
Step 04 — Your first LLM chain:  prompt | llm | parser
Created by Shivank Agarwal

Goal: send a reusable prompt (with {variables}) to the LLM and get plain text back.
The formulas are written inside the prompt and the LLM does the maths itself.
We also compute the exact answer in Python, so you can check whether the LLM got the maths right.

Run it:
    python 04_llm_chain.py
"""

import math
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(override=True)   # reads OPENAI_API_KEY (and optional OPENAI_MODEL) from .env
MODEL_ID = os.getenv("GROQ_MODEL_ID")
BASE_URL = os.getenv("GROQ_BASE_URL")
API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatOpenAI(model=MODEL_ID, reasoning_effort="none", base_url=BASE_URL, api_key=API_KEY)

# A prompt template = a message with blanks ({...}) that we fill in on every call
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an inventory planning expert.

Given these parameters, compute the desired stock level step by step.

Formulas:
  Safety Stock  = Z * demand_std_dev * sqrt(lead_time_days)
  Desired Stock = (avg_daily_demand * lead_time_days) + Safety Stock

Z-scores: 90% -> 1.28 | 95% -> 1.65 | 97% -> 1.88 | 98% -> 2.05 | 99% -> 2.33

End your response with exactly:
DESIRED_STOCK: <number>"""),
    ("human",
     "SKU: {sku_id} ({sku_name})\n"
     "Lead Time      : {lead_time_days} days\n"
     "Daily Demand   : {avg_daily_demand} units/day\n"
     "Demand Std Dev : {demand_std_dev} units/day\n"
     "Service Level  : {service_level}%"),
])

# ----- TODO 1: Build the chain: prompt -> llm -> StrOutputParser -----
# Hint: LangChain joins steps with the pipe operator |
# Hint: StrOutputParser() turns the LLM's message object into a plain string
# chain = None  # replace with your code
chain = prompt | llm | StrOutputParser()

Z_TABLE = {90: 1.28, 95: 1.65, 97: 1.88, 98: 2.05, 99: 2.33}

test_cases = [
    {"sku_id": "SKU-001", "sku_name": "Wireless Headphones", "lead_time_days": 21, "avg_daily_demand": 8,  "demand_std_dev": 2.0,  "service_level": 98},
    {"sku_id": "SKU-003", "sku_name": "Instant Noodles",     "lead_time_days": 4,  "avg_daily_demand": 80, "demand_std_dev": 10.0, "service_level": 95},
]

for params in test_cases:
    print(f"\n{'=' * 60}\nSKU: {params['sku_id']} — {params['sku_name']}\n{'=' * 60}")

    # ----- TODO 2: Run the chain with this SKU's parameters and print the result -----
    # Hint: chain.invoke(params) fills every {variable} in the prompt from the dict
    # result = "TODO"  # replace with your code
    result = chain.invoke(params)
    print(result)

    # The exact answer, computed in Python, to compare with the LLM's maths
    z = Z_TABLE[params["service_level"]]
    safety = z * params["demand_std_dev"] * math.sqrt(params["lead_time_days"])
    desired = params["avg_daily_demand"] * params["lead_time_days"] + safety
    print(f"\n[Python check] safety stock = {safety:.2f}, desired stock = {desired:.2f}")
