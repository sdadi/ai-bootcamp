"""
llm.py — one small wrapper around "an LLM", copied in spirit from the Deep Search course (deepsearch/llm.py).

Two modes:
* mock (default): no network. Every call site passes a `mock=` function that produces the answer with
  simple rules, so the whole project runs offline and the tests are repeatable.
* live: any OpenAI-compatible provider. Set LLM_PROVIDER in .env (see .env.example).

The method that matters is `structured()`: ask for JSON, validate it with a Pydantic schema, and if
validation fails send the error back and ask the model to repair its output. This validate → repair loop
is what turns fragile LLM output into a contract (Deep Search Module 4, Lab 2).
"""

from __future__ import annotations

import json
import re
from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .config import SETTINGS
from .tracing import Budget, Tracer, estimate_tokens

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> str:
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
    def __init__(self, budget: Budget | None = None, tracer: Tracer | None = None, model: str | None = None, role: str = "main"):
        self.budget = budget or Budget()
        self.tracer = tracer
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
                raise LLMError("Live mode needs the openai package: pip install openai") from exc
            self._client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=90, max_retries=2)

    @classmethod
    def small(cls, budget: Budget | None = None, tracer: Tracer | None = None) -> "LLM":
        """A cheaper model for simple, high-volume jobs such as claim extraction (model routing, Deep Search Module 9)."""
        return cls(budget=budget, tracer=tracer, model=SETTINGS.llm_small_model or None, role="small")

    # ------------------------------------------------------------------
    def chat(self, system: str, user: str, *, stage: str = "llm", json_mode: bool = False,
             mock: Callable[[], str] | None = None) -> str:
        if self.mock_mode:
            if mock is None:
                raise LLMError(f"Stage {stage!r} has no mock implementation. Set LLM_PROVIDER to a real provider or add a mock.")
            out = mock()
            self.budget.charge(estimate_tokens(system + user), estimate_tokens(out))
            if self.tracer:
                self.tracer.event("llm", llm_stage=stage, model=self.model, mock=True)
            return out

        kwargs: dict = {"model": self.model,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if SETTINGS.llm_temperature is not None:
            kwargs["temperature"] = SETTINGS.llm_temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — surface provider errors with a hint
            raise LLMError(f"LLM call failed at stage {stage!r}: {exc}") from exc
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) or estimate_tokens(system + user)
        out_tok = getattr(usage, "completion_tokens", None) or estimate_tokens(text)
        self.budget.charge(in_tok, out_tok)
        if self.tracer:
            self.tracer.event("llm", llm_stage=stage, model=self.model, input_tokens=in_tok, output_tokens=out_tok)
        return text

    def structured(self, system: str, user: str, schema: Type[T], *, stage: str = "llm",
                   mock: Callable[[], dict] | None = None, max_repairs: int = 2) -> T:
        """Ask for JSON matching `schema`; validate; repair on failure."""
        if self.mock_mode:
            if mock is None:
                raise LLMError(f"Stage {stage!r} has no mock implementation.")
            data = mock()
            self.budget.charge(estimate_tokens(system + user), estimate_tokens(json.dumps(data, default=str)))
            if self.tracer:
                self.tracer.event("llm", llm_stage=stage, model=self.model, mock=True, schema=schema.__name__)
            return schema.model_validate(data)

        schema_hint = json.dumps(schema.model_json_schema(), indent=0)
        sys_full = f"{system}\n\nReturn ONLY a JSON object that matches this JSON Schema:\n{schema_hint}"
        prompt = user
        last_err = ""
        for attempt in range(max_repairs + 1):
            raw = self.chat(sys_full, prompt, stage=f"{stage}#{attempt}", json_mode=True)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_err = str(exc)
                prompt = (f"{user}\n\nYour previous answer was not valid:\n{raw[:2000]}\n\n"
                          f"Validation error:\n{last_err[:1500]}\n\nReturn corrected JSON only.")
        raise LLMError(f"Could not get valid {schema.__name__} after {max_repairs} repairs: {last_err[:300]}")
