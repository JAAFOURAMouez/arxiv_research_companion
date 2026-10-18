"""
HTTP helper with bounded retries for OpenAI-compatible chat endpoints (Groq).

Retries on HTTP 429 and 5xx (500/502/503/504) and on connection errors / timeouts, with
exponential backoff + jitter.  A ``Retry-After`` header is honoured, but a server asking
for longer than ``max_delay`` is *not* waited on (a Streamlit user must not be blocked
for minutes): the error is raised immediately so the caller can degrade gracefully.
Other 4xx responses are never retried.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


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
    url: str = GROQ_CHAT_URL,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """POST an OpenAI-style chat completion and return the message content."""
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if model.startswith("openai/gpt-oss"):
        # reasoning models: keep reasoning short so tokens are left for the answer
        payload["reasoning_effort"] = "low"

    response = post_with_retry(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
        raise ValueError("Groq chat completion returned no choices")
    return (choices[0].get("message") or {}).get("content")
