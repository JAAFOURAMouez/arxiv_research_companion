"""
HTTP helper with bounded retries for OpenAI-compatible chat endpoints
(Ollama, Groq, OpenAI, OpenRouter; see ``PROVIDERS``).

Retries on HTTP 429 and 5xx (500/502/503/504) and on connection errors / timeouts, with
exponential backoff + jitter.  A ``Retry-After`` header is honoured, but a server asking
for longer than ``max_delay`` is *not* waited on (a Streamlit user must not be blocked
for minutes): the error is raised immediately so the caller can degrade gracefully.
Other 4xx responses are never retried.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True)
class Provider:
    """An OpenAI-compatible chat endpoint the app can talk to."""

    id: str
    label: str
    chat_url: str
    key_env: Optional[str]          # env var holding the server-side key; None = no key needed
    default_model: str
    token_param: str = "max_completion_tokens"
    local: bool = False

    @property
    def needs_key(self) -> bool:
        return self.key_env is not None


def ollama_base_url() -> str:
    """Where Ollama listens (OLLAMA_BASE_URL, e.g. http://host.docker.internal:11434)."""
    return (os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL).rstrip("/")


PROVIDERS: Dict[str, Provider] = {
    p.id: p
    for p in (
        Provider("ollama", "Ollama (local)", f"{DEFAULT_OLLAMA_URL}/v1/chat/completions", None,
                 "llama3.2", token_param="max_tokens", local=True),
        Provider("groq", "Groq", GROQ_CHAT_URL, "GROQ_API_KEY", "openai/gpt-oss-120b"),
        Provider("openai", "OpenAI", "https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY",
                 "gpt-4o-mini"),
        Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1/chat/completions",
                 "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct", token_param="max_tokens"),
    )
}


def chat_url(provider: str) -> str:
    """Chat endpoint of ``provider``; Ollama's honours OLLAMA_BASE_URL at call time."""
    if provider == "ollama":
        return f"{ollama_base_url()}/v1/chat/completions"
    return PROVIDERS[provider].chat_url


class LlmHttpError(RuntimeError):
    """Non-success HTTP result (after retries, where retrying applies)."""

    def __init__(self, message: str, status: Optional[int] = None, attempts: int = 1):
        super().__init__(message)
        self.status = status
        self.attempts = attempts


def _retry_after_seconds(response: Optional[requests.Response]) -> Optional[float]:
    if response is None:
        return None
    raw = response.headers.get("Retry-After") if response.headers else None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None  # HTTP-date form: ignore, fall back to exponential backoff


def post_with_retry(
    url: str,
    *,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
    timeout: float,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> requests.Response:
    last_error: Exception = RuntimeError("no attempt made")
    for attempt in range(1, max_attempts + 1):
        response: Optional[requests.Response] = None
        try:
            response = requests.post(
                url, headers=headers, json=json_payload, timeout=timeout
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = LlmHttpError(
                f"network error: {type(exc).__name__}", status=None, attempts=attempt
            )
        else:
            if response.ok:
                return response
            body = (response.text or "")[:300]
            last_error = LlmHttpError(
                f"HTTP {response.status_code}: {body}",
                status=response.status_code,
                attempts=attempt,
            )
            if response.status_code not in RETRYABLE_STATUS:
                raise last_error

        if attempt == max_attempts:
            break

        retry_after = _retry_after_seconds(response)
        if retry_after is not None:
            if retry_after > max_delay:
                raise LlmHttpError(
                    f"rate limited; server asked to wait {retry_after:.0f}s "
                    f"(> max_delay={max_delay:.0f}s)",
                    status=getattr(response, "status_code", None),
                    attempts=attempt,
                )
            delay = retry_after
        else:
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * (0.5 + rng())
        logger.warning(
            "LLM request failed (%s); retry %d/%d in %.1fs",
            last_error,
            attempt,
            max_attempts - 1,
            delay,
        )
        sleep(delay)

    if isinstance(last_error, LlmHttpError):
        last_error.attempts = max_attempts
    raise last_error


def chat_completion(
    messages: List[Dict[str, str]],
    *,
    model: str,
    api_key: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: Optional[float] = None,
    timeout: float = 60.0,
    url: Optional[str] = None,
    provider: str = "groq",
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """POST an OpenAI-style chat completion and return the message content."""
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"Unknown LLM provider: {provider}")
    if spec.needs_key and not api_key:
        raise ValueError(f"{spec.key_env} is not set")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        spec.token_param: max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if provider == "groq" and model.startswith("openai/gpt-oss"):
        # reasoning models: keep reasoning short so tokens are left for the answer
        payload["reasoning_effort"] = "low"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = post_with_retry(
        url or chat_url(provider),
        headers=headers,
        json_payload=payload,
        timeout=timeout,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        sleep=sleep,
    )
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"{spec.label} chat completion returned no choices")
    return (choices[0].get("message") or {}).get("content")
