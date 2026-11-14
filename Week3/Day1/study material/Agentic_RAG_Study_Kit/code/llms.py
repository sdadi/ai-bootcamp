"""
llms.py  -  Where we create the "brains" (LLMs)
================================================
Created by Shivank Agrawal

Our system uses TWO language models:

  1. MAIN LLM  (DeepSeek)  -> the brain of all 5 agents. It thinks, decides,
                              grades and writes the final answer.
  2. RAG LLM   (fast)      -> lives inside the RAG tool. Its only job is to read
                              the PDF chunks we found and write a short,
                              grounded answer from them. Speed matters here,
                              so by default it runs on Groq.

Think of a newsroom:
  - the MAIN LLM is the editor who decides what to publish,
  - the RAG LLM is a quick junior reporter who reads the files and
    hands the editor a short summary.
"""

import re
from functools import lru_cache

from crewai import LLM

import config


@lru_cache(maxsize=1)  # build it once, then reuse the same object
def get_main_llm() -> LLM:
    """The brain shared by every agent."""
    if config.LLM_MODE == "ollama":
        # DeepSeek running on YOUR laptop. Before running:
        #   1) install Ollama from ollama.com
        #   2) in a terminal:   ollama pull deepseek-r1:8b
        return LLM(
            model=f"ollama/{config.OLLAMA_MODEL}",
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.1,
        )

    if config.LLM_MODE == "groq":
        # Free-tier practice mode: a fast open model on Groq (not DeepSeek).
        return _groq_llm()

    # Default: DeepSeek in the cloud through OpenRouter.
    # The "openrouter/" prefix tells CrewAI which provider to call.
    return LLM(
        model=f"openrouter/{config.OPENROUTER_MODEL}",
        api_key=config.OPENROUTER_API_KEY,
        temperature=0.1,  # low temperature = calm, consistent answers
    )


def get_rag_llm() -> LLM:
    """The fast helper LLM used only inside the RAG tool."""
    if config.RAG_LLM_MODE == "groq":
        return _groq_llm()
    # "same" -> reuse the main brain
    return get_main_llm()


@lru_cache(maxsize=1)
def _groq_llm() -> LLM:
    # Groq speaks the same "language" (API format) as OpenAI,
    # so we use the OpenAI connector and just point it at Groq's address.
    return LLM(
        model=f"openai/{config.GROQ_MODEL}",
        base_url=config.GROQ_BASE_URL,
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
    )


def clean_text(text) -> str:
    """Reasoning models like DeepSeek-R1 sometimes print their private
    'thinking' inside <think> ... </think>. We only want the final answer."""
    text = str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()
