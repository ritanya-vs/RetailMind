"""A small, cached chat client for any OpenAI-compatible endpoint, with a fake for tests.

Everything that calls an LLM depends on the LLMClient protocol, never on a specific provider, so
the pipeline is testable without a key or a server. OpenAIChatClient talks to any OpenAI-style
/chat/completions endpoint: a local or tunnelled Ollama, Groq, and so on. It caches every response
on disk keyed by the exact request, so re-running evaluation does not repeat calls and stays
reproducible, and it retries transient network errors (useful for a flaky Kaggle tunnel).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": ...}

_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the LLM cannot be reached or returns nothing usable."""


class LLMClient(Protocol):
    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str: ...


def _cache_key(model: str, messages: list[Message], temperature: float, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    """Maps a request hash to the response text, one JSON file per entry."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def get(self, key: str) -> str | None:
        path = self.directory / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        return None

    def put(self, key: str, request: dict, response: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{key}.json").write_text(
            json.dumps({"request": request, "response": response}), encoding="utf-8"
        )


class OpenAIChatClient:
    """POSTs to {base_url}/chat/completions in the OpenAI chat format.

    base_url ends in the API root, e.g. 'http://localhost:11434/v1' for Ollama or
    'https://api.groq.com/openai/v1' for Groq. The api_key is sent as a bearer token; Ollama
    accepts any value. extra_headers lets a tunnel pass its own headers (e.g. ngrok's skip-warning).
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        if not model:
            raise LLMError("no LLM model configured; set llm.model or LLM_MODEL")
        if not base_url:
            raise LLMError("no LLM base_url configured; set llm.base_url or LLM_BASE_URL")
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self.extra_headers = dict(extra_headers or {})
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        key = _cache_key(self.model, messages, temperature, max_tokens)
        if self.cache and (hit := self.cache.get(key)) is not None:
            return hit
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        text = self._post_with_retries(body)
        if self.cache:
            self.cache.put(key, body, text)
        return text

    def _post_with_retries(self, body: dict) -> str:
        import requests  # already a dependency of ranx; imported lazily so tests need no network

        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.url, json=body, headers=self._headers(), timeout=self.timeout
                )
            except requests.RequestException as error:
                last_error = f"connection error: {error}"
            else:
                if response.status_code == 200:
                    return self._extract(response.json())
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in _TRANSIENT_STATUS:
                    break
            if attempt < self.max_retries:
                time.sleep(2**attempt)  # 1s, 2s, 4s backoff
        raise LLMError(
            f"request to {self.url} failed after {self.max_retries + 1} tries: {last_error}"
        )

    @staticmethod
    def _extract(payload: dict) -> str:
        try:
            return (payload["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"unexpected response shape: {error}") from error


def make_client(cfg, *, cache_dir: str | Path | None = None) -> OpenAIChatClient:
    """Build a client from an LLMConfig, letting env vars override the config."""
    model = os.environ.get("LLM_MODEL") or cfg.model
    base_url = os.environ.get("LLM_BASE_URL") or cfg.base_url
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get(cfg.api_key_env)
    return OpenAIChatClient(
        model,
        base_url,
        api_key=api_key,
        cache_dir=cache_dir,
        timeout=cfg.timeout_seconds,
        max_retries=cfg.max_retries,
    )


class ScriptedClient:
    """A fake LLM for tests: returns canned replies and records the prompts it received."""

    def __init__(self, replies: list[str] | None = None, default: str = "") -> None:
        self._replies = list(replies or [])
        self._default = default
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        self.calls.append(messages)
        return self._replies.pop(0) if self._replies else self._default
