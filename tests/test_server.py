"""
End-to-end tests of the web server (FastAPI TestClient, in-process).

No network and no model download: the ONNX embedder is replaced by a deterministic hash
embedder and the LLM HTTP call (Ollama by default) by a scripted fake.
"""
from datetime import datetime, timezone

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("chromadb")

from fastapi.testclient import TestClient  # noqa: E402

from arxiv_research_companion import data_ingestion as di  # noqa: E402
from arxiv_research_companion import llm_http, rag_engine, server  # noqa: E402
from arxiv_research_companion.embedders import HashEmbedder  # noqa: E402

PAPERS = [
    ("2610.00001", "Sparse attention", "Sparse attention reduces the memory cost of long context transformers."),
    ("2610.00002", "Diffusion models", "Diffusion models generate images by iterative denoising."),
    ("2610.00003", "Robot learning", "Reinforcement learning lets robots grasp objects with learned policies."),
    ("2610.00004", "Graph networks", "Graph neural networks propagate messages along the edges of a graph."),
]
QUESTION = "How does sparse attention reduce memory?"


class FakeOnnx:
    """Stands in for chromadb's ONNXMiniLM_L6_V2 (callable list[str] -> list[vector])."""

    def __init__(self, *a, **k):
        self._emb = HashEmbedder()

    def __call__(self, texts):
        return [row for row in self._emb.encode(list(texts))]


class Resp:
    def __init__(self, status=200, content="", headers=None):
        self.status_code, self.ok = status, status < 400
        self.headers = headers or {}
        self.text = f"status {status}"
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeLlm:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


@pytest.fixture
def env(tmp_path, monkeypatch):
    import chromadb.utils.embedding_functions as ef

    db = tmp_path / "papers.db"
    conn = di._init_db(str(db))
    for i, (pid, title, summary) in enumerate(PAPERS):
        di._store_paper(conn, {
            "arxiv_id": pid, "title": title, "authors": ["A. Author", "B. Writer"], "summary": summary,
            "categories": ["cs.AI"], "published": datetime(2026, 10, 1, 12, i, tzinfo=timezone.utc), "doi": None,
        })
    conn.close()

    monkeypatch.setattr(ef, "ONNXMiniLM_L6_V2", FakeOnnx)
    for key, value in {
        "INGESTION_DB_PATH": str(db), "ARC_CHROMA_DIR": str(tmp_path / "chroma"),
        "LLM_PROVIDER": "ollama", "LLM_MODEL_NAME": "llama3.2", "LLM_MAX_RETRIES": "2",
        "LLM_RETRY_BASE_DELAY": "0", "CITATIONS_ENFORCE": "true", "RETRIEVAL_MODE": "dense",
        "RATE_LIMIT_PER_MINUTE": "100",
    }.items():
        monkeypatch.setenv(key, value)
    return monkeypatch


def boot(app=None):
    """Start the app and wait for the background index build."""
    app = app or server.create_app()
    client = TestClient(app)
    client.__enter__()
    app.state.runtime.thread.join(timeout=60)
    assert app.state.runtime.ready, app.state.runtime.error
    return client


@pytest.fixture
def client(env):
    c = boot()
    yield c
    c.__exit__(None, None, None)


def ask(client, question=QUESTION, **extra):
    return client.post("/api/ask", json={"question": question, **extra})


def test_page_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "arXiv Research Companion" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}


def test_status_reports_the_corpus(client):
    status = client.get("/api/status").json()
    assert status["ready"] is True
    assert status["papers"] == len(PAPERS)
    assert status["last_paper_date"] == "2026-10-01"
    assert status["llm_configured"] is True
    assert status["retrieval_mode"] == "dense"


def test_not_ready_is_reported_and_refuses_questions(env):
    def never_ready(self):
        pass

    env.setattr(server.Runtime, "load", never_ready)
    with TestClient(server.create_app()) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/api/status").json() == {"ready": False, "error": None}
        assert ask(c).status_code == 503


def test_answer_with_citations_and_sources(env, client):
    env.setattr(llm_http.requests, "post", FakeLlm(Resp(content="Sparse attention lowers memory cost for long inputs [1].")))
    data = ask(client).json()
    assert "lowers memory cost" in data["answer"]
    assert not data["degraded"] and not data["refused"]
    first = data["sources"][0]
    assert first["n"] == 1 and first["title"] == "Sparse attention" and first["cited"] is True
    assert first["url"] == "https://arxiv.org/abs/2610.00001"
    assert first["authors"] == ["A. Author", "B. Writer"]
    assert first["published"] == "2026-10-01"
    assert not any(s["cited"] for s in data["sources"][1:])


def test_k_controls_the_number_of_sources(env, client):
    env.setattr(llm_http.requests, "post", FakeLlm(Resp(content="Sparse attention lowers memory cost for long inputs [1].")))
    assert len(ask(client, k=2).json()["sources"]) == 2
    assert ask(client, k=0).status_code == 422
    assert ask(client, k=11).status_code == 422


def test_llm_5xx_still_shows_sources(env, client):
    fake = FakeLlm(Resp(503))
    env.setattr(llm_http.requests, "post", fake)
    data = ask(client).json()
    assert fake.calls == 3  # 1 try + 2 retries (LLM_MAX_RETRIES=2)
    assert data["degraded"] is True
    assert "failed to generate an answer" in data["answer"]
    assert "Here are the papers" not in data["answer"]  # the page lists them itself
    assert data["sources"], "sources must be returned even though the LLM failed"


def test_429_then_success_is_retried_transparently(env, client):
    fake = FakeLlm(Resp(429), Resp(content="Diffusion models denoise images step by step [1]."))
    env.setattr(llm_http.requests, "post", fake)
    data = ask(client, "How do diffusion models generate images?").json()
    assert fake.calls == 2
    assert not data["degraded"]
    assert "denoise images" in data["answer"]


def test_refusal_is_shown_with_closest_sources(env, client):
    env.setattr(llm_http.requests, "post", FakeLlm(Resp(content="NOT_ENOUGH_INFORMATION")))
    data = ask(client, "What is the capital of France?").json()
    assert data["refused"] is True
    assert "don't contain enough information" in data["answer"]
    assert data["sources"]


def test_uncited_sentences_are_removed_and_reported(env, client):
    answer = "Sparse attention lowers memory cost [1]. This extra claim carries no citation at all."
    env.setattr(llm_http.requests, "post", FakeLlm(Resp(content=answer)))
    data = ask(client).json()
    assert "extra claim" not in data["answer"]
    assert data["dropped_count"] == 1


def test_generation_crash_falls_back_to_sources_only(env, client):
    def boom(*a, **k):
        raise RuntimeError("unexpected")

    env.setattr(rag_engine, "generate_answer", boom)
    data = ask(client).json()
    assert data["degraded"] is True
    assert data["sources"]
    assert "unexpected" not in data["answer"]  # internal errors are not shown to visitors


def test_bad_questions_are_rejected(client):
    assert ask(client, "").status_code == 422
    assert ask(client, "   ").status_code == 422
    assert ask(client, "x" * (server.MAX_QUESTION_CHARS + 1)).status_code == 422


def test_rate_limit_per_client(env):
    env.setenv("RATE_LIMIT_PER_MINUTE", "2")
    env.setattr(llm_http.requests, "post", FakeLlm(Resp(content="Sparse attention lowers memory cost for long inputs [1].")))
    c = boot()
    try:
        codes = [ask(c).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        other = c.post("/api/ask", json={"question": QUESTION}, headers={"x-forwarded-for": "203.0.113.9"})
        assert other.status_code == 200
    finally:
        c.__exit__(None, None, None)


# --------------------------------------------------------------------------- #
# Provider / model picker                                                     #
# --------------------------------------------------------------------------- #
class Capture(FakeLlm):
    """FakeLlm that also remembers the URL, headers and payload of each call."""

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.last = {"url": url, "headers": headers, "json": json}
        return super().__call__(url, headers=headers, json=json, timeout=timeout)


GOOD = "Sparse attention lowers memory cost for long inputs [1]."


def test_default_provider_is_ollama_without_key(env, client):
    cap = Capture(Resp(content=GOOD))
    env.setattr(llm_http.requests, "post", cap)
    data = ask(client).json()
    assert not data["degraded"]
    assert cap.last["url"] == "http://localhost:11434/v1/chat/completions"
    assert "Authorization" not in cap.last["headers"]
    assert cap.last["json"]["model"] == "llama3.2"
    assert "max_tokens" in cap.last["json"]


def test_status_reports_provider(client):
    status = client.get("/api/status").json()
    assert status["provider"] == "ollama" and status["llm_configured"] is True


def test_providers_lists_choices_without_leaking_keys(env, client):
    env.setenv("GROQ_API_KEY", "server-secret")
    data = client.get("/api/providers").json()
    ids = [p["id"] for p in data["providers"]]
    assert {"ollama", "groq", "openai", "openrouter", "huggingface_api"} <= set(ids)
    assert data["default"] == {"provider": "ollama", "model": "llama3.2"}
    assert "server-secret" not in str(data)
    ollama = next(p for p in data["providers"] if p["id"] == "ollama")
    assert ollama["needs_key"] is False and ollama["local"] is True


def test_user_key_and_model_reach_the_chosen_provider(env, client):
    cap = Capture(Resp(content=GOOD))
    env.setattr(llm_http.requests, "post", cap)
    choice = {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-user"}
    data = ask(client, llm=choice).json()
    assert not data["degraded"]
    assert cap.last["url"] == "https://api.openai.com/v1/chat/completions"
    assert cap.last["headers"]["Authorization"] == "Bearer sk-user"
    assert cap.last["json"]["model"] == "gpt-4o-mini"


def test_paid_provider_without_a_key_is_refused(env, client):
    cap = Capture(Resp(content=GOOD))
    env.setattr(llm_http.requests, "post", cap)
    r = ask(client, llm={"provider": "groq", "model": "llama-3.3-70b-versatile"})
    assert r.status_code == 422 and "API key" in r.json()["detail"]
    assert cap.calls == 0


def test_unknown_provider_is_refused(client):
    assert ask(client, llm={"provider": "nope", "model": "x"}).status_code == 422


def test_server_key_is_not_spent_on_other_models(env):
    env.setenv("LLM_PROVIDER", "groq")
    env.setenv("GROQ_API_KEY", "server-secret")
    env.setenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")
    c = boot()
    try:
        cap = Capture(Resp(content=GOOD))
        env.setattr(llm_http.requests, "post", cap)
        # the server's own provider + model: allowed without a key from the visitor
        ok = ask(c, llm={"provider": "groq", "model": "openai/gpt-oss-120b"})
        assert ok.status_code == 200 and cap.last["headers"]["Authorization"] == "Bearer server-secret"
        # another model on the owner's key: refused
        assert ask(c, llm={"provider": "groq", "model": "llama-3.3-70b-versatile"}).status_code == 422
        # the same with the visitor's own key: fine
        own = ask(c, llm={"provider": "groq", "model": "llama-3.3-70b-versatile", "api_key": "gsk-own"})
        assert own.status_code == 200 and cap.last["headers"]["Authorization"] == "Bearer gsk-own"
    finally:
        c.__exit__(None, None, None)


def test_ollama_unreachable_gives_a_helpful_message(env, client):
    def refuse(*a, **k):
        raise llm_http.requests.ConnectionError("refused")

    env.setattr(llm_http.requests, "post", refuse)
    env.setenv("LLM_RETRY_BASE_DELAY", "0")
    env.setattr(llm_http.time, "sleep", lambda s: None)
    data = ask(client).json()
    assert data["degraded"] and "ollama serve" in data["answer"]


def test_ollama_models_endpoint(env, client):
    class R:
        def raise_for_status(self): pass
        def json(self): return {"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.2:latest"}]}

    env.setattr(server.requests, "get", lambda url, timeout=None: R())
    assert client.get("/api/ollama/models").json() == {
        "reachable": True, "models": ["llama3.2:latest", "qwen2.5:7b"]}

    def down(url, timeout=None):
        raise server.requests.ConnectionError("down")

    env.setattr(server.requests, "get", down)
    assert client.get("/api/ollama/models").json() == {"reachable": False, "models": []}
