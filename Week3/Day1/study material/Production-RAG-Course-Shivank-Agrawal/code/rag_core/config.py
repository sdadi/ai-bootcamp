"""
Settings for the whole course, read from environment variables (or a .env file).

Copy .env.example to .env and edit it. Every setting has a safe default, so the labs
run with NO API keys at all (a local embedding model + an "extractive" answer mode).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the "code" folder

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_float(name: str, default: float | None) -> float | None:
    value = _env(name)
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Default model names. Check your provider's model page before you change these:
# model names get deprecated. These were current in November 2026.
DEFAULT_LLM_MODEL = {
    "openai": "gpt-6-luna",          # cheap, fast OpenAI model (replaces gpt-4.1-mini / gpt-4o-mini)
    "anthropic": "claude-haiku-5-5",  # cheap, fast Anthropic model
    "ollama": "llama3.2",            # any model you have pulled locally with `ollama pull`
}


def _auto_provider() -> str:
    explicit = _env("LLM_PROVIDER").lower()
    if explicit:
        return explicit
    if _env("OPENAI_API_KEY"):
        return "openai"
    if _env("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"  # no LLM: the labs fall back to an extractive answer


@dataclass
class Settings:
    # ---- embeddings -------------------------------------------------------
    # fastembed  = local ONNX models, downloaded once from Hugging Face (default)
    # openai     = OpenAI embeddings API (needs OPENAI_API_KEY)
    # wordllama  = tiny model shipped inside the pip package (works fully offline)
    # hashing    = toy keyword "embedding" with no model at all (for tests)
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "fastembed").lower())
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    # ---- reranking --------------------------------------------------------
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))

    # ---- LLM --------------------------------------------------------------
    llm_provider: str = field(default_factory=_auto_provider)  # none | openai | anthropic | ollama
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    llm_temperature: float | None = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", None))

    # ---- retrieval behaviour ---------------------------------------------
    # If the best vector similarity is below this, we refuse to answer.
    # None = no threshold. Lab 09 shows you how to pick a value for YOUR model.
    min_similarity: float | None = field(default_factory=lambda: _env_float("MIN_SIMILARITY", None))

    # ---- folders ----------------------------------------------------------
    data_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    log_dir: Path = ROOT / "logs"
    reports_dir: Path = ROOT / "reports"

    def __post_init__(self) -> None:
        if not self.llm_model and self.llm_provider in DEFAULT_LLM_MODEL:
            self.llm_model = DEFAULT_LLM_MODEL[self.llm_provider]
        for d in (self.processed_dir, self.log_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s
