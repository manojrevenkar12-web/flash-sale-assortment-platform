"""Minimal, provider-agnostic chat client with timeouts, retries and usage accounting.

Providers: "anthropic" (Messages API) and "openai" (any OpenAI-compatible chat completions
endpoint, e.g. Azure OpenAI, Bedrock gateways, vLLM). Configured from the environment:

    FSAP_LLM_PROVIDER=anthropic|openai   FSAP_LLM_MODEL=...   FSAP_LLM_API_KEY=...
    FSAP_LLM_BASE_URL=...  (optional)

With no provider configured the curation step runs on the deterministic lexicon extractor and
template copy generator, which is what CI and the demo use.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: list[float] = field(default_factory=list)


class LLMClient:
    model: str

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 20.0,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ):
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"unknown provider {provider}")
        self.provider, self.model, self.retries = provider, model, retries
        default = "https://api.anthropic.com" if provider == "anthropic" else "https://api.openai.com"
        headers = (
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            if provider == "anthropic"
            else {"Authorization": f"Bearer {api_key}"}
        )
        self.http = httpx.Client(base_url=base_url or default, headers=headers, timeout=timeout, transport=transport)
        self.usage = Usage()

    @classmethod
    def from_env(cls) -> LLMClient | None:
        provider = os.environ.get("FSAP_LLM_PROVIDER")
        if not provider:
            return None
        return cls(
            provider, os.environ["FSAP_LLM_MODEL"], os.environ["FSAP_LLM_API_KEY"], os.environ.get("FSAP_LLM_BASE_URL")
        )

    def complete(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
        if self.provider == "anthropic":
            path, body = (
                "/v1/messages",
                {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        else:
            path, body = (
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = time.perf_counter()
            try:
                r = self.http.post(path, json=body)
                if r.status_code in (429, 500, 502, 503, 529):
                    raise LLMError(f"retryable status {r.status_code}")
                if r.status_code >= 400:
                    self.usage.failures += 1
                    raise LLMError(f"status {r.status_code}: {r.text[:200]}")
                data = r.json()
                self.usage.calls += 1
                self.usage.latency_ms.append((time.perf_counter() - t0) * 1000)
                if self.provider == "anthropic":
                    self.usage.input_tokens += data.get("usage", {}).get("input_tokens", 0)
                    self.usage.output_tokens += data.get("usage", {}).get("output_tokens", 0)
                    return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
                self.usage.input_tokens += data.get("usage", {}).get("prompt_tokens", 0)
                self.usage.output_tokens += data.get("usage", {}).get("completion_tokens", 0)
                return data["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.TransportError, LLMError) as e:
                last = e
                if isinstance(e, LLMError) and "retryable" not in str(e):
                    raise
                time.sleep(min(2.0, 0.25 * 2**attempt))
        self.usage.failures += 1
        raise LLMError(f"gave up after {self.retries + 1} attempts: {last}")


class ScriptedLLM(LLMClient):
    """Test double: answers from a function of the prompt, records calls. Used to exercise the
    guardrails with the failure modes real models show (invented materials, prose, bad JSON)."""

    def __init__(self, respond: Callable[[str], str], model: str = "scripted"):
        self.model, self.provider = model, "scripted"
        self.respond = respond
        self.usage = Usage()
        self.prompts: list[str] = []

    def complete(self, prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        self.usage.calls += 1
        out = self.respond(prompt)
        if isinstance(out, Exception):
            self.usage.failures += 1
            raise out
        return out
