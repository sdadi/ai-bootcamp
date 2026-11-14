"""
Chapter 9 - One tiny client for several LLM providers.

    LLM_PROVIDER=openai     uses OPENAI_API_KEY (also works with any OpenAI-compatible server
                            by setting OPENAI_BASE_URL, e.g. Groq, OpenRouter, vLLM, LM Studio)
    LLM_PROVIDER=anthropic  uses ANTHROPIC_API_KEY
    LLM_PROVIDER=ollama     a local model served by Ollama (free, private, slower on laptops)
    LLM_PROVIDER=none       no LLM: labs fall back to an "extractive" answer (quotes the best sentences)

Every call returns the text, token counts and latency, because in production you must
track cost and speed for every request.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass

from .config import Settings, get_settings


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""


class LLM:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.provider = self.s.llm_provider
        self.model = self.s.llm_model
        self._client = None
        if self.provider in {"openai", "ollama"}:
            from openai import OpenAI

            if self.provider == "ollama":
                self._client = OpenAI(base_url=self.s.openai_base_url or "http://localhost:11434/v1", api_key="ollama")
            else:
                self._client = OpenAI(base_url=self.s.openai_base_url or None)
        elif self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic()
        elif self.provider != "none":
            raise ValueError(f"Unknown LLM_PROVIDER={self.provider!r}")

    @property
    def available(self) -> bool:
        return self._client is not None

    def __repr__(self) -> str:
        return f"LLM(provider={self.provider}, model={self.model})" if self.available else "LLM(none)"

    def generate(self, system: str, user: str, max_tokens: int = 700, json_mode: bool = False) -> LLMResult:
        if not self.available:
            raise RuntimeError("No LLM configured. Set OPENAI_API_KEY / ANTHROPIC_API_KEY or LLM_PROVIDER=ollama.")
        t0 = time.perf_counter()
        if self.provider == "anthropic":
            if json_mode:
                user += "\n\nReply with valid JSON only. No text before or after the JSON."
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content)
            return LLMResult(text.strip(), resp.usage.input_tokens, resp.usage.output_tokens,
                             (time.perf_counter() - t0) * 1000, self.model)
        kwargs = {"model": self.model,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.s.llm_temperature is not None:
            # Many newer reasoning models only accept the default temperature, so we send it
            # only when you set LLM_TEMPERATURE explicitly.
            kwargs["temperature"] = self.s.llm_temperature
        resp = self._client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        return LLMResult((resp.choices[0].message.content or "").strip(),
                         getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0,
                         (time.perf_counter() - t0) * 1000, self.model)

    def read_image(self, png_bytes: bytes, instruction: str = "Transcribe all text in this image. "
                   "Keep tables as Markdown tables. Output only the transcription.") -> LLMResult:
        """VLM-based OCR: send a page image to a vision-capable model (Chapter 5).
        Costs money per page, so route only hard pages here (tables, handwriting, bad scans)."""
        if not self.available:
            raise RuntimeError("No LLM configured.")
        b64 = base64.b64encode(png_bytes).decode()
        t0 = time.perf_counter()
        if self.provider == "anthropic":
            resp = self._client.messages.create(model=self.model, max_tokens=2000, messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": instruction}]}])
            text = "".join(getattr(b, "text", "") for b in resp.content)
            return LLMResult(text, resp.usage.input_tokens, resp.usage.output_tokens,
                             (time.perf_counter() - t0) * 1000, self.model)
        resp = self._client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}])
        u = resp.usage
        return LLMResult(resp.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens,
                         (time.perf_counter() - t0) * 1000, self.model)
