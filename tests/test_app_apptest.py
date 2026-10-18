"""
End-to-end UI tests with Streamlit's AppTest (runs streamlit_app.py in-process).

No network and no model download: the ONNX embedder is replaced by a deterministic hash
embedder and the Groq HTTP call by a scripted fake.  Requires `streamlit` and `chromadb`
(both runtime dependencies); skipped if they are missing.

NOTE: written without being able to execute it in the authoring sandbox (no PyPI). If an
AppTest accessor differs in your Streamlit version, adjust the helpers at the top.
"""
import os
import sys

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("chromadb")

import numpy as np  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, "streamlit_app.py")

from arxiv_research_companion import data_ingestion as di  # noqa: E402
from arxiv_research_companion import llm_http  # noqa: E402
from arxiv_research_companion.embedders import HashEmbedder  # noqa: E402

PAPERS = [
    ("2610.00001", "Sparse attention", "Sparse attention reduces the memory cost of long context transformers."),
    ("2610.00002", "Diffusion models", "Diffusion models generate images by iterative denoising."),
    ("2610.00003", "Robot learning", "Reinforcement learning lets robots grasp objects with learned policies."),
    ("2610.00004", "Graph networks", "Graph neural networks propagate messages along the edges of a graph."),
]


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


class FakeGroq:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


@pytest.fixture
def env(tmp_path, monkeypatch):
    import chromadb.utils.embedding_functions as ef
    import streamlit as st
    from datetime import datetime, timezone

    db = tmp_path / "papers.db"
    conn = di._init_db(str(db))
    for i, (pid, title, summary) in enumerate(PAPERS):
        di._store_paper(conn, {
            "arxiv_id": pid, "title": title, "authors": ["A. Author"], "summary": summary,
            "categories": ["cs.AI"], "published": datetime(2026, 10, 1, 12, i, tzinfo=timezone.utc), "doi": None,
        })
    conn.close()

    monkeypatch.setattr(ef, "ONNXMiniLM_L6_V2", FakeOnnx)
    for key, value in {
        "INGESTION_DB_PATH": str(db), "ARC_CHROMA_DIR": str(tmp_path / "chroma"),
        "GROQ_API_KEY": "test-key", "LLM_MAX_RETRIES": "2", "LLM_RETRY_BASE_DELAY": "0",
        "CITATIONS_ENFORCE": "true", "RETRIEVAL_MODE": "dense",
    }.items():
        monkeypatch.setenv(key, value)
    st.cache_resource.clear()
    yield monkeypatch
    st.cache_resource.clear()


def ask(at, question):
    at.chat_input[0].set_value(question).run(timeout=60)
    return at


def boot():
    return AppTest.from_file(APP, default_timeout=60).run()


def texts(at):
    return [m.value for m in at.markdown]


def expander_labels(at):
    return [e.label for e in at.expander]


def test_app_boots_and_shows_database_status(env):
    at = boot()
    assert not at.exception
    assert [m.value for m in at.sidebar.metric] == [str(len(PAPERS))]


def test_answer_with_citations_and_sources(env):
    fake = FakeGroq(Resp(content="Sparse attention lowers memory cost for long inputs [1]."))
    env.setattr(llm_http.requests, "post", fake)
    at = ask(boot(), "How does sparse attention reduce memory?")
    assert not at.exception
    assert any("lowers memory cost" in t for t in texts(at))
    labels = expander_labels(at)
    assert labels and labels[0].startswith("[1] Sparse attention")
    assert "cited" in labels[0]


def test_llm_5xx_still_shows_sources(env):
    fake = FakeGroq(Resp(503))
    env.setattr(llm_http.requests, "post", fake)
    at = ask(boot(), "How does sparse attention reduce memory?")
    assert not at.exception
    assert fake.calls == 3  # 1 try + 2 retries (LLM_MAX_RETRIES=2)
    assert any("could not be generated" in w.value for w in at.warning)
    assert any("failed to generate an answer" in t for t in texts(at))
    assert expander_labels(at), "sources must be rendered even though the LLM failed"


def test_429_then_success_is_retried_transparently(env):
    fake = FakeGroq(Resp(429), Resp(content="Diffusion models denoise images step by step [1]."))
    env.setattr(llm_http.requests, "post", fake)
    at = ask(boot(), "How do diffusion models generate images?")
    assert not at.exception
    assert fake.calls == 2
    assert not list(at.warning)
    assert any("denoise images" in t for t in texts(at))


def test_refusal_is_shown_with_closest_sources(env):
    env.setattr(llm_http.requests, "post", FakeGroq(Resp(content="NOT_ENOUGH_INFORMATION")))
    at = ask(boot(), "What is the capital of France?")
    assert not at.exception
    assert any("don't contain enough information" in t for t in texts(at))
    assert expander_labels(at)


def test_uncited_sentences_are_removed_and_reported(env):
    answer = "Sparse attention lowers memory cost [1]. This extra claim carries no citation at all."
    env.setattr(llm_http.requests, "post", FakeGroq(Resp(content=answer)))
    at = ask(boot(), "How does sparse attention reduce memory?")
    assert not at.exception
    assert not any("extra claim" in t for t in texts(at) if "removed" not in t)
    assert any("without a valid citation were removed" in c.value for c in at.caption)


def test_generation_crash_falls_back_to_sources_only(env):
    import arxiv_research_companion.rag_engine as rag

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    env.setattr(rag, "generate_answer", boom)  # streamlit_app imports the name at run time
    at = ask(boot(), "How does sparse attention reduce memory?")
    assert not at.exception
    assert expander_labels(at)
