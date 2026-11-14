import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from strands import Agent
from strands.models.openai import OpenAIModel

load_dotenv(override=True)   # reads OPENAI_API_KEY (and optional OPENAI_MODEL) from .env

MODEL_ID = os.getenv("GROQ_MODEL_ID")
API_KEY = os.getenv("GROQ_API_KEY", "")
BASE_URL = os.getenv("GROQ_BASE_URL")
model = OpenAIModel(
    client_args={
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "max_retries": 5,  # retry dropped connections (the client default is 2)
    },
    model_id=MODEL_ID,
    # Groq reasoning models stream their thinking by default; hide it so only the answer prints.
    params={"extra_body": {"include_reasoning": False}},
)

agent = Agent(model=model,
              system_prompt=("You are a helpful assistant that helps with fixing the coding issues basedon the debugging information."))

response = agent("Based on the following debugging information, what is the most likely cause of the error? and what is the best way to fix it?\n\nTraceback (most recent call last):\n  File \"main.py\", line 10, in <module>\n    import non_existent_module\nModuleNotFoundError: No module named 'non_existent_module'")