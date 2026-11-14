"""
config.py — every setting in one place, read from .env.

Same pattern as the Deep Search course (deepsearch/config.py) and the Agentic RAG kit (config.py):
copy .env.example to .env, edit, and nothing else in the code needs to change.

Defaults are OFFLINE: LLM_PROVIDER=mock and SEARCH_PROVIDER=offline. The whole project runs with no
API key and no internet. Switch providers only after the offline run is green.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is in requirements.txt; this keeps `import assessor` working without it
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"
VENDOR_CORPUS = DATA_DIR / "vendor_web" / "corpus.json"
REQUIREMENTS_FILE = DATA_DIR / "requirements.yaml"
GOLD_SET = DATA_DIR / "gold_set.json"
RUNS_DIR = ROOT / "runs"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # --- LLM -------------------------------------------------------------
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "mock").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_small_model: str = field(default_factory=lambda: _env("LLM_SMALL_MODEL", ""))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))

    # --- Search --------------------------------------------------------------
    search_provider: str = field(default_factory=lambda: _env("SEARCH_PROVIDER", "offline").lower())
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY", ""))
    results_per_query: int = field(default_factory=lambda: _env_int("VRA_RESULTS_PER_QUERY", 5))

    # --- Agent behaviour -----------------------------------------------------
    as_of: date = field(default_factory=lambda: date.fromisoformat(_env("VRA_AS_OF", "2026-11-01")))
    max_usd: float = field(default_factory=lambda: _env_float("VRA_MAX_USD", 0.50))
    max_parallel: int = field(default_factory=lambda: _env_int("VRA_MAX_PARALLEL", 4))
    verbose: bool = field(default_factory=lambda: _env("VRA_VERBOSE", "1") == "1")

    # --- Cost estimates (only used for the budget maths) --------------------
    price_in_per_mtok: float = field(default_factory=lambda: _env_float("PRICE_IN_PER_MTOK", 0.10))
    price_out_per_mtok: float = field(default_factory=lambda: _env_float("PRICE_OUT_PER_MTOK", 0.50))

    @property
    def offline(self) -> bool:
        return self.llm_provider == "mock"

    def resolved_llm(self) -> dict:
        """Translate LLM_PROVIDER into (base_url, api_key, model) for the OpenAI-compatible client.

        LEARNER NOTE: this table is copied from the Deep Search course (deepsearch/config.py). If a model
        name here is out of date, pick a current one from the provider's models page and set LLM_MODEL.
        """
        p = self.llm_provider
        table = {
            "openai":    ("https://api.openai.com/v1", _env("OPENAI_API_KEY"), "gpt-6-luna"),
            "gemini":    ("https://generativelanguage.googleapis.com/v1beta/openai/", _env("GEMINI_API_KEY"), "gemini-3.8-flash"),
            "groq":      ("https://api.groq.com/openai/v1", _env("GROQ_API_KEY"), "openai/gpt-oss-20b"),
            "ollama":    ("http://localhost:11434/v1", "ollama", "qwen3:8b"),
            "anthropic": ("https://api.anthropic.com/v1/", _env("ANTHROPIC_API_KEY"), "claude-haiku-5-5"),
            "openrouter": ("https://openrouter.ai/api/v1", _env("OPENROUTER_API_KEY"), "deepseek/deepseek-chat"),
            "custom":    (_env("LLM_BASE_URL", "http://localhost:8000/v1"), _env("LLM_API_KEY", "none"), ""),
        }
        if p not in table:
            raise ValueError(f"Unknown LLM_PROVIDER={p!r}. Use mock, openai, gemini, groq, ollama, anthropic, openrouter or custom.")
        base_url, api_key, default_model = table[p]
        if not api_key:
            raise ValueError(f"LLM_PROVIDER={p} needs an API key in .env (see .env.example).")
        return {"base_url": base_url, "api_key": api_key, "model": self.llm_model or default_model}


SETTINGS = Settings()
