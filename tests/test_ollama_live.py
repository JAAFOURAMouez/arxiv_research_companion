"""
Tests against a real, local Ollama. Skipped unless an Ollama server answers and has the model.

    ollama serve &                 # if it is not already running
    ollama pull llama3.2           # or set OLLAMA_TEST_MODEL to a model you already have
    uv run pytest -m ollama

OLLAMA_BASE_URL / OLLAMA_TEST_MODEL override the address (default http://localhost:11434)
and the model (default llama3.2). Small models are weak writers, so the assertions only check
that the plumbing works (a non-empty answer, sources present), never the wording.
"""
import os

import pytest
import requests

pytest.importorskip("fastapi")
pytest.importorskip("chromadb")

from arxiv_research_companion.llm_http import chat_completion, ollama_base_url  # noqa: E402
from tests.test_server import ask, boot, env  # noqa: E402,F401  (env is a fixture)

pytestmark = pytest.mark.ollama

MODEL = os.getenv("OLLAMA_TEST_MODEL", "llama3.2")


@pytest.fixture(scope="module")
def ollama_model():
    try:
        tags = requests.get(f"{ollama_base_url()}/api/tags", timeout=2).json()
    except (requests.RequestException, ValueError):
        pytest.skip(f"no Ollama server at {ollama_base_url()}")
    names = [m.get("name", "") for m in tags.get("models", [])]
    match = next((n for n in names if n == MODEL or n.split(":")[0] == MODEL.split(":")[0]), None)
    if match is None:
        pytest.skip(f"model '{MODEL}' is not pulled (ollama pull {MODEL}); have: {names}")
    return match


def test_chat_completion_roundtrip(ollama_model):
    out = chat_completion(
        [{"role": "user", "content": "Reply with the single word: pong"}],
        model=ollama_model, api_key=None, max_tokens=20, temperature=0.0,
        timeout=180, provider="ollama", max_attempts=1,
    )
    assert isinstance(out, str) and out.strip()


def test_server_answers_with_a_real_local_model(env, ollama_model):
    env.setenv("LLM_PROVIDER", "ollama")
    env.setenv("LLM_MODEL_NAME", ollama_model)
    env.setenv("CITATIONS_ENFORCE", "false")  # a 3B model will not reliably cite
    client = boot()
    try:
        data = ask(client).json()
    finally:
        client.__exit__(None, None, None)
    assert data["sources"]
    assert not data["degraded"], data["answer"]
    assert data["answer"].strip()


def test_picker_choice_selects_the_local_model(env, ollama_model):
    env.setenv("CITATIONS_ENFORCE", "false")
    client = boot()
    try:
        data = ask(client, llm={"provider": "ollama", "model": ollama_model}).json()
    finally:
        client.__exit__(None, None, None)
    assert not data["degraded"], data["answer"]
