"""
llm.py — one small wrapper around "an LLM".

Two modes:

* mock  (default) — no network. Each call site passes a `mock=` function that
  produces the answer with simple rules. The wrapper still estimates tokens and
  charges the Budget, so cost/latency lessons stay meaningful offline.
* live  — any OpenAI-compatible provider (OpenAI, Gemini, Groq, Ollama,
  Anthropic's compatibility layer, OpenRouter...). Set LLM_PROVIDER in .env.

The most important method is `structured()`: ask for JSON, validate it with a
Pydantic schema, and if validation fails, send the error back to the model and
ask it to repair its output (up to `max_repairs` times). This "validate →
repair" loop is what turns fragile LLM output into a reliable contract.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .budget import Budget, estimate_tokens
from .config import SETTINGS
from .tracing import trace

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add a sentence. Strip that."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    if text.startswith("{") or text.startswith("["):
        return text
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        return text[first : last + 1]
    return text


class LLM:
    def __init__(self, budget: Budget | None = None, model: str | None = None, role: str = "main"):
        self.budget = budget or Budget()
        self.provider = SETTINGS.llm_provider
        self.mock_mode = self.provider == "mock"
        self.role = role
        self._client = None
        if self.mock_mode:
            self.model = "mock-llm"
        else:
            cfg = SETTINGS.resolved_llm()
            self.model = model or cfg["model"]
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise LLMError("pip install openai") from exc
            self._client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=90, max_retries=2)

    @classmethod
    def small(cls, budget: Budget | None = None) -> "LLM":
        """A cheaper model for simple, high-volume jobs (model routing)."""
        return cls(budget=budget, model=SETTINGS.llm_small_model or None, role="small")

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, *, stage: str = "other", json_mode: bool = False,
             mock: Callable[[], str] | None = None) -> str:
        """Plain text in, plain text out."""
        started = time.time()
        if self.mock_mode:
            if mock is None:
                raise LLMError(f"No mock provided for stage '{stage}'. Set LLM_PROVIDER to a real provider.")
            out = mock()
            in_tok, out_tok = estimate_tokens(system + user), estimate_tokens(out)
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            kwargs = {"model": self.model, "messages": messages}
            if os.getenv("LLM_TEMPERATURE"):  # many reasoning models only accept their default
                kwargs["temperature"] = float(os.getenv("LLM_TEMPERATURE"))
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # some providers reject response_format — retry without it
                if json_mode and "response_format" in str(exc).lower():
                    kwargs.pop("response_format", None)
                    resp = self._client.chat.completions.create(**kwargs)
                else:
                    raise LLMError(f"{self.provider} call failed: {exc}") from exc
            out = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", None) or estimate_tokens(system + user)
            out_tok = getattr(usage, "completion_tokens", None) or estimate_tokens(out)
        self.budget.charge_llm(self.model, in_tok, out_tok, stage=stage)
        trace("llm_call", stage=stage, model=self.model, input_tokens=in_tok, output_tokens=out_tok,
              seconds=round(time.time() - started, 3))
        return out

    # ------------------------------------------------------------------
    def structured(self, system: str, user: str, schema: Type[T], *, stage: str = "other",
                   mock: Callable[[], T] | None = None, max_repairs: int = 2) -> T:
        """Ask for JSON matching `schema`; validate; repair on failure."""
        if self.mock_mode:
            if mock is None:
                raise LLMError(f"No mock provided for stage '{stage}'.")
            obj = mock()
            self.chat(system, user, stage=stage, mock=lambda: obj.model_dump_json())  # charge tokens
            return schema.model_validate(obj.model_dump())

        schema_json = json.dumps(schema.model_json_schema(), indent=None)
        sys_prompt = (
            f"{system}\n\nReturn ONLY one JSON object that matches this JSON Schema. "
            f"No prose, no markdown fences.\nJSON Schema: {schema_json}"
        )
        prompt = user
        last_error = ""
        for attempt in range(max_repairs + 1):
            raw = self.chat(sys_prompt, prompt, stage=stage, json_mode=True)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:1500]
                trace("schema_repair", stage=stage, attempt=attempt + 1, error=last_error[:300])
                prompt = (
                    f"{user}\n\nYour previous answer was:\n{raw[:3000]}\n\n"
                    f"It failed validation with this error:\n{last_error}\n"
                    "Return a corrected JSON object only."
                )
        raise LLMError(f"Could not get valid {schema.__name__} after {max_repairs} repairs: {last_error}")

    # ------------------------------------------------------------------
    def tool_call_round(self, messages: list[dict], tools: list[dict], stage: str = "tools"):
        """One round of native function calling (used in Lab 2, live mode only)."""
        if self.mock_mode:
            raise LLMError("tool_call_round needs a live provider")
        kwargs = {"model": self.model, "messages": messages, "tools": tools}
        if os.getenv("LLM_REASONING_EFFORT"):  # e.g. gpt-6-luna needs "none" for function calling
            kwargs["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT")
        resp = self._client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        self.budget.charge_llm(self.model, getattr(usage, "prompt_tokens", 0) or 0,
                               getattr(usage, "completion_tokens", 0) or 0, stage=stage)
        return resp.choices[0].message
