"""
config.py — every setting the project reads, in one place.

Settings come from environment variables (or a `.env` file in the project
root). If you set nothing at all, the project runs fully OFFLINE:

    LLM_PROVIDER=mock        -> rule-based stand-ins instead of a real LLM
    SEARCH_PROVIDER=offline  -> searches the small fictional web in data/corpus.json

That is deliberate: you can learn the whole pipeline without an API key,
then switch on a real LLM and real web search when you are ready.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:  # python-dotenv is optional; if present we load .env automatically
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover - fine to run without it
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

# ---------------------------------------------------------------------------
# LLM provider presets. Every provider below speaks the OpenAI "chat
# completions" format, so ONE client (the `openai` Python SDK) talks to all of
# them — you only change the base URL, the key and the model name.
# Model names change often (defaults checked Nov 2026). If a default stops
# working, open the provider's model list and put a current name in LLM_MODEL.
# ---------------------------------------------------------------------------
PROVIDER_PRESETS: dict[str, dict[str, str | None]] = {
    "openai": {
        "base_url": None,  # SDK default: https://api.openai.com/v1
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-6-luna",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-3.8-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-20b",
    },
    "ollama": {  # runs on your laptop, free, no key needed
        "base_url": "http://localhost:11434/v1",
        "key_env": "",
        "default_model": "qwen3:8b",
    },
    "anthropic": {  # Anthropic's OpenAI-SDK compatibility layer
        "base_url": "https://api.anthropic.com/v1/",
        "key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-5-5",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "openai/gpt-oss-120b",
    },
    "custom": {  # any other OpenAI-compatible server
        "base_url": os.getenv("LLM_BASE_URL"),
        "key_env": "LLM_API_KEY",
        "default_model": "",
    },
}


def _parse_date(value: str | None, fallback: str) -> date:
    return date.fromisoformat(value or fallback)


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock").lower())
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    llm_base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    # A cheaper model for simple jobs (claim extraction). Empty = same as llm_model.
    llm_small_model: str = field(default_factory=lambda: os.getenv("LLM_SMALL_MODEL", ""))
    search_provider: str = field(default_factory=lambda: os.getenv("SEARCH_PROVIDER", "offline").lower())
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    # "Today" for freshness maths. Offline mode pins it so results are repeatable.
    as_of: date = field(
        default_factory=lambda: _parse_date(
            os.getenv("DS_AS_OF"),
            "2026-11-01" if os.getenv("SEARCH_PROVIDER", "offline").lower() == "offline" else date.today().isoformat(),
        )
    )
    max_iterations: int = field(default_factory=lambda: int(os.getenv("DS_MAX_ITERATIONS", "3")))
    max_parallel: int = field(default_factory=lambda: int(os.getenv("DS_MAX_PARALLEL", "4")))
    results_per_query: int = field(default_factory=lambda: int(os.getenv("DS_RESULTS_PER_QUERY", "3")))
    verbose: bool = field(default_factory=lambda: os.getenv("DS_VERBOSE", "1") == "1")

    # ------------------------------------------------------------------
    def resolved_llm(self) -> dict[str, str | None]:
        """Return base_url / api_key / model for the chosen provider."""
        if self.llm_provider == "mock":
            return {"base_url": None, "api_key": "mock", "model": "mock-llm"}
        if self.llm_provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.llm_provider}'. "
                f"Use one of: mock, {', '.join(PROVIDER_PRESETS)}"
            )
        preset = PROVIDER_PRESETS[self.llm_provider]
        key_env = preset["key_env"] or ""
        api_key = self.llm_api_key or (os.getenv(key_env, "") if key_env else "") or "not-needed"
        if self.llm_provider not in ("ollama",) and api_key == "not-needed":
            raise RuntimeError(
                f"LLM_PROVIDER={self.llm_provider} needs an API key. Put it in .env as "
                f"{key_env or 'LLM_API_KEY'}=... (see .env.example), or use LLM_PROVIDER=mock."
            )
        model = self.llm_model or preset["default_model"]
        if not model:
            raise RuntimeError("Set LLM_MODEL in .env for the 'custom' provider.")
        return {"base_url": self.llm_base_url or preset["base_url"], "api_key": api_key, "model": model}


SETTINGS = Settings()


def describe() -> str:
    """One line you can print at the top of every lab."""
    s = SETTINGS
    llm = "mock (offline, rule-based)" if s.llm_provider == "mock" else f"{s.llm_provider}:{s.resolved_llm()['model']}"
    return f"LLM={llm} | search={s.search_provider} | as_of={s.as_of.isoformat()}"
