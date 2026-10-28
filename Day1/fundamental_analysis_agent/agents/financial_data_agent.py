import os

from dotenv import load_dotenv
from strands import Agent
from strands.models.openai import OpenAIModel

load_dotenv(override=True)  # reads GROQ_API_KEY from a .env file, if present

MODEL_ID = os.getenv("GROQ_LLM_MODEL")
BASE_URL = os.getenv('GROQ_BASE_URL')
API_KEY = os.getenv('GROQ_API_KEY')

print(f"Base url: {BASE_URL}")
print(f"Model id: {MODEL_ID}")
print(f"LLM Model id: {MODEL_ID}")
print(f"API Key: {API_KEY}")

model = OpenAIModel(
    client_args={
        "api_key": API_KEY,
        "base_url": BASE_URL,
        "max_retries": 5,  # retry dropped connections (the client default is 2)
    },
    model_id=MODEL_ID,
    # Groq reasoning models stream their thinking by deaulft; hide it so only the answer prints.
    # params={"extra_body": {"include_reasoning": False}},
)

print(f"Using {MODEL_ID} on Groq")
