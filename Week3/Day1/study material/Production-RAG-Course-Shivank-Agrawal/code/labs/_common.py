"""Shared setup for the labs: lets `python labs/xx.py` import rag_core from the code folder,
and gives friendly errors when a model cannot be downloaded."""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
os.chdir(CODE_DIR)
warnings.filterwarnings("ignore", category=UserWarning, module="rag_core")

try:  # quieter logs from fastembed's downloader
    from loguru import logger

    logger.remove()
except Exception:
    pass


def embedder_or_exit():
    """Load the embedding model from .env, or explain how to fix the most common problem."""
    from rag_core.embeddings import get_embedder

    try:
        return get_embedder()
    except Exception as e:
        print("\n[!] Could not load the embedding model:", e.__class__.__name__, str(e)[:200])
        print("    Most likely your network blocks huggingface.co (common on office networks).")
        print("    Fix: run with the offline model instead, for example:")
        print("        EMBEDDER=wordllama python labs/<lab>.py      (macOS / Linux)")
        print("        set EMBEDDER=wordllama && python labs/<lab>.py   (Windows cmd)")
        print("    or put EMBEDDER=wordllama in your .env file.")
        sys.exit(1)


def llm_status():
    from rag_core.llm import LLM

    llm = LLM()
    if llm.available:
        print(f"LLM: {llm.provider} / {llm.model}")
    else:
        print("LLM: none (no API key found). Answers use the free 'extractive' fallback.")
        print("     Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env, or run Ollama, to see real LLM answers.")
    return llm


def default_groups(item: dict) -> list[str]:
    return item.get("user_groups", ["public", "employees"])
