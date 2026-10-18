import unittest
from unittest.mock import patch

import requests

from arxiv_research_companion import llm_http
from arxiv_research_companion.llm_http import LlmHttpError, chat_completion, post_with_retry


class Resp:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code = status
        self.ok = status < 400
        self._body = body or {}
        self.headers = headers or {}
        self.text = text or str(body)

    def json(self):
        return self._body


OK = Resp(200, {"choices": [{"message": {"content": "hello"}}]})


def script(*items):
    """requests.post replacement that yields responses / raises exceptions in order."""
    it = iter(items)
    calls = []

    def fake(url, headers=None, json=None, timeout=None):
        calls.append(json)
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    fake.calls = calls
    return fake


class TestRetry(unittest.TestCase):
    def run_chat(self, fake, **kw):
        sleeps = []
        with patch.object(llm_http.requests, "post", fake):
            try:
                out = chat_completion(
                    [{"role": "user", "content": "hi"}], model="m", api_key="k",
                    max_tokens=10, temperature=0.0, sleep=sleeps.append, **kw)
            except Exception as exc:  # noqa: BLE001
                return exc, sleeps, fake
        return out, sleeps, fake

    def test_429_then_success_retries(self):
        out, sleeps, fake = self.run_chat(script(Resp(429), OK))
        self.assertEqual(out, "hello")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(len(sleeps), 1)

    def test_each_retryable_status(self):
        for status in (429, 500, 502, 503, 504):
            out, _, fake = self.run_chat(script(Resp(status), OK))
            self.assertEqual(out, "hello", status)

    def test_non_retryable_4xx_fails_immediately(self):
        for status in (400, 401, 403, 404):
            out, sleeps, fake = self.run_chat(script(Resp(status, text="nope"), OK))
            self.assertIsInstance(out, LlmHttpError)
            self.assertEqual(out.status, status)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(sleeps, [])

    def test_gives_up_after_max_attempts(self):
        out, sleeps, fake = self.run_chat(script(*[Resp(503)] * 4), max_attempts=4)
        self.assertIsInstance(out, LlmHttpError)
        self.assertEqual((out.status, out.attempts), (503, 4))
        self.assertEqual(len(fake.calls), 4)
        self.assertEqual(len(sleeps), 3)

    def test_retry_after_header_is_honoured(self):
        out, sleeps, _ = self.run_chat(script(Resp(429, headers={"Retry-After": "2"}), OK))
        self.assertEqual(out, "hello")
        self.assertEqual(sleeps, [2.0])

    def test_retry_after_longer_than_cap_fails_fast(self):
        out, sleeps, fake = self.run_chat(script(Resp(429, headers={"Retry-After": "600"}), OK))
        self.assertIsInstance(out, LlmHttpError)
        self.assertEqual(out.status, 429)
        self.assertEqual(sleeps, [])
        self.assertEqual(len(fake.calls), 1)

    def test_backoff_grows_exponentially_without_header(self):
        sleeps = []
        with patch.object(llm_http.requests, "post", script(Resp(500), Resp(500), Resp(500), OK)):
            post_with_retry("u", headers={}, json_payload={}, timeout=1, max_attempts=4,
                            base_delay=1.0, max_delay=100, sleep=sleeps.append, rng=lambda: 0.5)
        self.assertEqual(sleeps, [1.0, 2.0, 4.0])  # rng=0.5 -> jitter factor 1.0

    def test_connection_error_and_timeout_retry(self):
        out, _, fake = self.run_chat(script(requests.ConnectionError("x"), requests.Timeout("y"), OK))
        self.assertEqual(out, "hello")
        self.assertEqual(len(fake.calls), 3)

    def test_missing_key(self):
        with self.assertRaises(ValueError):
            chat_completion([], model="m", api_key="", max_tokens=1, temperature=0)

    def test_gpt_oss_gets_reasoning_effort(self):
        _, _, fake = self.run_chat(script(OK))
        self.assertNotIn("reasoning_effort", fake.calls[0])
        sleeps = []
        fake2 = script(OK)
        with patch.object(llm_http.requests, "post", fake2):
            chat_completion([{"role": "user", "content": "x"}], model="openai/gpt-oss-120b",
                            api_key="k", max_tokens=5, temperature=0.1, sleep=sleeps.append)
        self.assertEqual(fake2.calls[0]["reasoning_effort"], "low")


if __name__ == "__main__":
    unittest.main()
