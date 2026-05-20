"""Thin wrapper over an OpenAI-compatible chat API (DeepSeek by default)."""
from __future__ import annotations

import time

import httpx
from openai import OpenAI

from .config import Config


class LLMClient:
    def __init__(self, config: Config) -> None:
        if not config.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        # trust_env=False so httpx ignores HTTP_PROXY/HTTPS_PROXY; the LLM
        # endpoint is internal and any system proxy (e.g. Squid) would block it.
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
            http_client=httpx.Client(trust_env=False, timeout=config.request_timeout),
        )
        self._model = config.model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    def complete(self, system: str, user: str, retries: int = 3) -> str:
        """Single chat completion with exponential-backoff retries."""
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - surfaced after the retry budget
                last_err = e
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM request failed after {retries} attempts: {last_err}")
