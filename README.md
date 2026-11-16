# arXiv Research Companion

An AI-powered research assistant that helps users quickly find and synthesize insights from academic papers using Retrieval-Augmented Generation (RAG) over arXiv.org.

## Features

- Fetches papers from arXiv API (categories: cs.AI, cs.LG, stat.ML, etc.)
- Creates semantic embeddings with all-MiniLM-L6-v2 (ONNX, no PyTorch)
- Stores embeddings in a vector database (ChromaDB)
- Retrieves relevant paper abstracts for user queries
- Generates grounded answers with an LLM of your choice: local Ollama (no key), Groq, OpenAI, OpenRouter or Hugging Face. The page has a **Model** section to pick provider and model, and to paste your own API key (sent with the question, never stored on the server)
- Every answer sentence cites the retrieved paper it comes from as `[n]`; uncited sentences are repaired once, then removed
- Retrieval modes: dense, BM25, hybrid, hybrid + cross-encoder rerank (chosen from the evaluation in `eval/`)
- Incremental updates: only processes new papers since last run
- Web interface: a plain HTML/CSS/JS page over a small FastAPI API
- Designed to run on CPU-only environments (WSL2/Linux compatible)
- Uses only free APIs and open-source tools

## Architecture

The system follows a clean, layered architecture:

1. **Data Ingestion Layer**: Fetches and parses arXiv papers, handles deduplication
2. **Embedding & Vector Storage Layer**: Generates embeddings and manages similarity search
3. **RAG Engine Layer**: Retrieves relevant context and generates answers with LLMs
4. **Presentation Layer**: FastAPI server and the static page in `web/`
5. **Configuration & Utilities**: Centralized config, logging, and helper functions

## Getting Started

### Prerequisites

- Python 3.12 (pinned in `.python-version` / `pyproject.toml`) and [uv](https://docs.astral.sh/uv/)
- Git
- (Optional) GPU for faster embeddings (CPU-only works fine)

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:JAAFOURAMouez/arxiv_research_companion.git
   cd arxiv_research_companion
   ```

2. Install dependencies (exactly what `uv.lock` pins):
   ```bash
   uv sync --locked
   ```

3. Choose how the model runs: copy `.env.example` to `.env`, then either
   * **local and free:** install [Ollama](https://ollama.com), `ollama pull llama3.2`, set `LLM_PROVIDER=ollama`; or
   * **hosted:** fill in `GROQ_API_KEY` (or `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_API_TOKEN`).

   You can also pick the provider, model and your own key later, in the page itself (see [Models](#models-local-ollama-or-your-own-api-key)).

### Usage

```bash
uv run uvicorn arxiv_research_companion.server:app --reload
```

Then open http://localhost:8000. The page is plain HTML/CSS/JS in `web/`, served by a small
FastAPI app (`arxiv_research_companion/server.py`) that exposes:

| route | purpose |
|---|---|
| `GET /api/status` | index ready?, paper count, default provider/model |
| `POST /api/ask` | `{question, k, llm?}`; `llm` = `{provider, model, api_key}` overrides the default for this question |
| `GET /api/providers` | providers and default models for the picker (never returns a key) |
| `GET /api/ollama/models` | models found on the Ollama server |
| `GET /healthz` | liveness check |

The previous Streamlit front-end (`uv run streamlit run streamlit_app.py`) still works and will be
removed once the Render deployment is live.

### Deploying on Render (free plan)

`render.yaml` describes the service. In the Render dashboard: **New > Blueprint**, pick this
repository, then type `GROQ_API_KEY` when asked. Render then:

* builds with `pip install -r requirements.txt` and embeds every paper ahead of time
  (`python -m arxiv_research_companion.app_state`), so a cold start only loads the index;
* starts `uvicorn` on `$PORT` and checks `/healthz`;
* redeploys on every push to the tracked branch.

Measured locally: about 380 MB peak RAM (free plan: 512 MB). Embedding runs without the ONNX
memory arena and in batches of 8 (`ONNX_BATCH_SIZE`); the default chromadb settings peaked
around 760 MB. Free instances sleep after 15 minutes idle, so the first visit after that waits
for a restart. `RATE_LIMIT_PER_MINUTE` (default 10) caps questions per visitor, since every
question spends your server key's quota (visitors using their own key do not).

## Project Structure

```
arxiv_research_companion/
├── web/                               # Front-end: index.html, styles.css, app.js
├── render.yaml                        # Render deployment
├── streamlit_app.py                   # previous Streamlit front-end (to be removed)
├── arxiv_research_companion/          # Main package
│   ├── data_ingestion.py              # arXiv API ingestion (SQLite)
│   ├── ingest_cli.py                  # CLI used by the nightly ingestion workflow
│   ├── embedders.py                   # ONNX / sentence-transformers embedders
│   ├── embedding_store.py             # Vector storage (ChromaDB)
│   ├── retrieval.py                   # dense / BM25 / hybrid / rerank retrieval
│   ├── rag_engine.py                  # RAG pipeline: retrieve + generate
│   ├── server.py                      # FastAPI: JSON API + static front-end
│   ├── app_state.py                   # builds config, index and retriever (also the build step)
│   ├── citations.py                   # [n] citation validation, repair and refusal detection
│   ├── llm_http.py                    # OpenAI-compatible client (Ollama/Groq/OpenAI/OpenRouter) with retries
│   ├── interface.py, main.py          # legacy Gradio interface
│   ├── config.py                      # Configuration management
│   └── utils.py                       # Helper functions
├── eval/                              # Evaluation scripts, run_all.sh, labelling tool (see eval/README.md)
├── tests/                             # Unit, API and (optional) live-Ollama tests
├── arxiv_papers.db                    # Paper database, updated nightly by CI
├── pyproject.toml, uv.lock            # Dependencies (source of truth)
└── requirements.txt                   # Exported from uv.lock (installed by Render)
```

## Configuration

Configuration is managed via:
- Environment variables (loaded via python-dotenv)
- Default values in `config.py`

Key configurable parameters:
- arXiv categories to follow
- Embedding model name
- Vector database path and settings
- LLM provider and model (`LLM_PROVIDER`: ollama, groq, openai, openrouter or huggingface_api; `LLM_MODEL_NAME` defaults to a sensible model for the provider)
- Ingestion schedule
- Interface settings (port, theme, etc.)

### Hugging Face Inference Providers

With `LLM_PROVIDER=huggingface_api` (or when no other key is set), the default model is `openai/gpt-oss-120b`, but it can only run if your Hugging Face account has enabled a compatible Inference Provider and your token can make Inference Providers calls. If the app reports that no provider supports the model, enable a compatible provider in your Hugging Face account settings or choose a model supported by a provider you have enabled. A model being available on Hugging Face does not guarantee that your account can call it through an Inference Provider.

To pin routing to a provider enabled for your account, set `HF_INFERENCE_PROVIDER` to its provider ID. If unset, Hugging Face selects automatically from providers enabled for the account. Keep `LLM_PROVIDER` as `huggingface_api`; it selects the application backend, not a Hugging Face inference provider.

## Dependencies

See `requirements.txt` for the full list. Key packages include:
- `fastapi`, `uvicorn`
- `chromadb`
- `numpy`
- `requests`
- `python-dotenv`
- `huggingface_hub`
- `pydantic`

Evaluation-only packages (`ranx`, `sentence-transformers`, `ragas`, ...) live in the `eval` group and are never installed on the deployed app. `gradio` is a dev dependency for the legacy interface tests.

## Future Enhancements

- Full-text processing (PDF download and section-aware chunking)
- User personalization (saved searches, topic alerts)
- A retrieval-score gate for out-of-corpus questions (the evaluation shows top-1 cosine separates them well)
- Multimodal support (figure/table extraction)
- Collaboration features (shared annotations, exports)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- arXiv.org for providing open access to scientific papers
- Hugging Face for open-source models and inference API
- Ollama for making local models easy to run
- The developers of sentence-transformers, ChromaDB, and FastAPI

## Development (uv, tests, CI)

```bash
make lock                      # after changing pyproject.toml: updates uv.lock and requirements.txt (Render installs it)
uv sync --locked               # install exactly what is locked
uv run pytest -m "not slow and not network"   # unit + API tests (no network; live Ollama tests skip themselves)
make test-ollama               # live tests against a real local Ollama
uv run uvicorn arxiv_research_companion.server:app --reload
```

* GitHub Actions: `ci.yml` (locked install, requirements/lock drift check, pytest incl. AppTest) and
  `nightly-ingest.yml` (daily arXiv ingestion, commits `arxiv_papers.db` when it changed).
* LLM calls retry on HTTP 429/5xx and network errors with backoff; if generation still fails the retrieved papers are shown, with a hint (Ollama not running, model not pulled, bad key...).
* Answers must cite retrieved papers as `[n]`; uncited sentences are repaired once, then removed. A refusal is `NOT_ENOUGH_INFORMATION`; capitalised variants small models produce (`NOT ENOUGH INFORMATION`) are accepted too.
* `RETRIEVAL_MODE` = `dense` (default, unchanged behaviour) | `bm25` | `hybrid` | `hybrid_rerank`.
  Choose it from the numbers produced by the evaluation scripts, see [`eval/README.md`](eval/README.md).

## Models: local Ollama or your own API key

Pick the provider and model in the **Model** section of the page, or set a default with
`LLM_PROVIDER` / `LLM_MODEL_NAME` (see `.env.example`).

* **Ollama (local, free):** `ollama serve`, `ollama pull llama3.2`, then choose *Ollama*. The app
  must run where Ollama is reachable (`OLLAMA_BASE_URL`, default `http://localhost:11434`), so a
  hosted deployment on Render cannot reach the Ollama on your laptop.
* **Your own key:** choose Groq / OpenAI / OpenRouter / Hugging Face and paste a key. It is sent with
  that question only, never stored or logged by the server, and kept in your browser tab (cleared when
  it closes). The server's own key is only used for the server's default provider and model; any other
  paid choice needs your key (otherwise the server answers with an error). Ollama needs no key.

### Tests and evaluation with Ollama

* `uv run pytest` fakes the HTTP call (Ollama is the default provider in `tests/test_server.py`).
* `make test-ollama` (`pytest -m ollama`) runs live tests against a real Ollama; they skip when no
  server or model is found. `OLLAMA_TEST_MODEL` picks the model (default `llama3.2`).
* **Evaluation:** `bash eval/run_all.sh` runs on Ollama by default (generator `llama3.2`, judge
  `qwen2.5:7b`, two different families on purpose) and checks that the server and both models are
  there before starting. For Groq: `LLM_PROVIDER=groq GROQ_API_KEY=... bash eval/run_all.sh`.
  Small local models cite and refuse less reliably than large hosted ones, so generation figures
  (citations, refusal, faithfulness) are not comparable across providers; retrieval figures do not
  involve an LLM. The 30 calibration pairs are labelled by hand, in a spreadsheet or in the terminal
  with `python -m eval.label_calibration`. Details in [`eval/README.md`](eval/README.md).
