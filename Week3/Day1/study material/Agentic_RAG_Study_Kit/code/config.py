"""
config.py  -  The "settings room" of our project
=================================================
Created by Shivank Agrawal

Every secret key and every model name lives here, in ONE place.
The other files simply do:   import config

Why?  Think of the main switchboard in your house. You don't rewire every
room to turn off the lights - you flip one switch. Same here: change a model
in ONE place, and the whole project follows.

The actual secret values are NOT written in this file.
They are read from a file called `.env` (copy `.env.example` -> `.env`).
Never share your `.env` file or push it to GitHub.
"""

import os

from dotenv import load_dotenv

# Read the .env file and put every line into "environment variables".
load_dotenv()

# ---------------------------------------------------------------------------
# 1) API KEYS  (see the study guide for where to get each one)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # DeepSeek in the cloud
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")              # very fast cloud models
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")          # web search

# ---------------------------------------------------------------------------
# 2) WHICH "BRAIN" DO THE 5 AGENTS USE?
#    "openrouter" -> DeepSeek in the cloud (default). Pay-as-you-go, but a
#                    question costs a tiny fraction of a rupee.
#    "groq"       -> a fast open model on Groq's free tier (not DeepSeek).
#                    Good for practising without any payment.
#    "ollama"     -> DeepSeek running on your own laptop. Free and private,
#                    but slow, and needs about 8 GB of free RAM.
# ---------------------------------------------------------------------------
LLM_MODE = os.getenv("LLM_MODE", "openrouter").strip().lower()

# DeepSeek model on OpenRouter. It must support "tools" (function calling),
# because our Retriever agent calls tools. Check openrouter.ai/models.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")

# DeepSeek on your laptop via Ollama  (run:  ollama pull deepseek-r1:8b)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# 3) THE SECOND, FAST LLM - used only inside the RAG tool to turn the
#    retrieved PDF chunks into a short answer.
#    "groq"  -> very fast cloud inference (needs GROQ_API_KEY)
#    "same"  -> reuse the main brain above (no Groq key needed)
# ---------------------------------------------------------------------------
RAG_LLM_MODE = os.getenv("RAG_LLM_MODE", "groq").strip().lower()

# One Groq model name, used for the RAG LLM (and for the brain if LLM_MODE=groq)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# ---------------------------------------------------------------------------
# 4) EMBEDDINGS - the model that turns text into numbers (vectors).
#    Runs locally on your laptop. Downloaded once from Hugging Face.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ---------------------------------------------------------------------------
# 5) OUR KNOWLEDGE BASE (the private document)
# ---------------------------------------------------------------------------
PDF_PATH = os.getenv("PDF_PATH", "data/attention_is_all_you_need.pdf")
PDF_URL = os.getenv("PDF_URL", "https://arxiv.org/pdf/1706.03762")

# One line that tells the Router agent WHAT is inside the PDF.
# The router never reads the PDF itself - it decides using this description.
# If you swap in your own PDF, update this line too!
PDF_TOPIC = os.getenv(
    "PDF_TOPIC",
    "the Transformer neural-network architecture: attention, self-attention, "
    "multi-head attention, positional encoding, encoder-decoder design and "
    "how it was trained for machine translation",
)

# ---------------------------------------------------------------------------
# 6) SAFETY LIMITS
# ---------------------------------------------------------------------------
WEB_RESULTS = int(os.getenv("WEB_RESULTS", "3"))   # how many web results to fetch
TOP_K = int(os.getenv("TOP_K", "4"))               # how many PDF chunks to retrieve
MAX_RPM = int(os.getenv("MAX_RPM", "10"))          # max LLM calls per minute (free tiers are strict)
ANSWERS_LOG = os.getenv("ANSWERS_LOG", "answers_log.md")  # every Q&A is saved here


def check_keys() -> list[str]:
    """Return a list of friendly warnings for settings or keys that are missing."""
    problems = []
    if LLM_MODE not in ("openrouter", "groq", "ollama"):
        problems.append(f"LLM_MODE='{LLM_MODE}' is not valid. Use openrouter, groq or ollama.")
    if RAG_LLM_MODE not in ("groq", "same"):
        problems.append(f"RAG_LLM_MODE='{RAG_LLM_MODE}' is not valid. Use groq or same.")
    if LLM_MODE == "openrouter" and not OPENROUTER_API_KEY:
        problems.append("OPENROUTER_API_KEY is empty (or choose LLM_MODE=groq / ollama).")
    if (LLM_MODE == "groq" or RAG_LLM_MODE == "groq") and not GROQ_API_KEY:
        problems.append("GROQ_API_KEY is empty (or set RAG_LLM_MODE=same).")
    if not TAVILY_API_KEY:
        problems.append("TAVILY_API_KEY is empty - web search will not work.")
    return problems
