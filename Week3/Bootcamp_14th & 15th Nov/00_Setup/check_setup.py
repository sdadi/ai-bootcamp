"""
Setup check: run this once BEFORE the session
Created by Shivank Agarwal

    python 00_Setup/check_setup.py

It checks your libraries, downloads the free local models now (so class time is not spent downloading),
and makes two tiny OpenAI calls to confirm your API key works. Cost: a tiny fraction of a rupee.
"""

import importlib
import os
import sys


def ok(msg):
    print(f"  [OK]   {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


print(f"Python {sys.version.split()[0]}")
if sys.version_info < (3, 10):
    fail("Python 3.10 or newer is needed (3.11 - 3.13 recommended)")
    sys.exit(1)

# 1) Libraries --------------------------------------------------------------
print("\n1) Libraries")
missing = []
for name in ["openai", "langchain_core", "langchain_openai", "langchain_text_splitters", "langchain_experimental",
             "langchain_chroma", "langchain_huggingface", "chromadb", "sentence_transformers", "gensim",
             "rank_bm25", "gradio", "sklearn", "pandas", "dotenv"]:
    try:
        importlib.import_module(name)
        ok(name)
    except Exception as e:
        fail(f"{name}: {e}")
        missing.append(name)
if missing:
    print("\nInstall the missing libraries with:  pip install -r 00_Setup/requirements.txt")
    sys.exit(1)

# 2) Free local models -----------------------------------------------------
print("\n2) Free local models (the first time this takes a few minutes)")
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
ok("all-MiniLM-L6-v2 (sentence embeddings)")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
ok("ms-marco-MiniLM-L6-v2 (reranker)")

import gensim.downloader as api  # noqa: E402

api.load("glove-wiki-gigaword-50")
ok("glove-wiki-gigaword-50 (word vectors)")

# 3) OpenAI ------------------------------------------------------------------
print("\n3) OpenAI API key")
from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # looks for a .env file in this folder and in the folders above it
if not os.getenv("OPENAI_API_KEY"):
    fail("OPENAI_API_KEY not found. Create the .env file in the main folder (see the setup guide).")
    sys.exit(1)

from openai import OpenAI  # noqa: E402

client = OpenAI()
try:
    reply = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[{"role": "user", "content": "Reply with just the word OK"}],
        reasoning_effort="none",
    )
    ok(f"chat model gpt-6-luna replied: {reply.choices[0].message.content!r}")
    emb = client.embeddings.create(model="text-embedding-3-small", input="hello")
    ok(f"embedding model text-embedding-3-small returned {len(emb.data[0].embedding)} numbers")
except Exception as e:
    fail(f"OpenAI call failed: {e}")
    sys.exit(1)

print("\nAll good. You are ready for the session!")
