# arXiv Research Companion

An AI-powered research assistant that helps users quickly find and synthesize insights from academic papers using Retrieval-Augmented Generation (RAG) over arXiv.org.

## Features

- Fetches papers from arXiv API (categories: cs.AI, cs.LG, stat.ML, etc.)
- Creates semantic embeddings using sentence-transformers
- Stores embeddings in a vector database (ChromaDB)
- Retrieves relevant paper abstracts for user queries
- Generates grounded answers using LLMs (via Hugging Face Inference API)
- Provides citations to source papers
- Incremental updates: only processes new papers since last run
- Web interface built with Gradio for easy interaction
- Designed to run on CPU-only environments (WSL2/Linux compatible)
- Uses only free APIs and open-source tools

## Architecture

The system follows a clean, layered architecture:

1. **Data Ingestion Layer**: Fetches and parses arXiv papers, handles deduplication
2. **Embedding & Vector Storage Layer**: Generates embeddings and manages similarity search
3. **RAG Engine Layer**: Retrieves relevant context and generates answers with LLMs
4. **Presentation Layer**: Gradio-based web interface for user interaction
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

3. Set your API key: copy `secrets.toml.example` to `.streamlit/secrets.toml` and fill in `GROQ_API_KEY`.

### Usage

```bash
uv run streamlit run streamlit_app.py
```

Then open the URL shown in the output (typically http://localhost:8501).

The legacy Gradio front-end is still available with `uv run python -m arxiv_research_companion.main`.

## Project Structure

```
arxiv_research_companion/
├── streamlit_app.py                   # Streamlit app (deployed entry point)
├── arxiv_research_companion/          # Main package
│   ├── data_ingestion.py              # arXiv API ingestion (SQLite)
│   ├── ingest_cli.py                  # CLI used by the nightly ingestion workflow
│   ├── embedders.py                   # ONNX / sentence-transformers embedders
│   ├── embedding_store.py             # Vector storage (ChromaDB)
│   ├── retrieval.py                   # dense / BM25 / hybrid / rerank retrieval
│   ├── rag_engine.py                  # RAG pipeline: retrieve + generate
│   ├── citations.py                   # [n] citation validation and repair
│   ├── llm_http.py                    # Groq HTTP client with retries
│   ├── interface.py, main.py          # legacy Gradio interface
│   ├── config.py                      # Configuration management
│   └── utils.py                       # Helper functions
├── eval/                              # Evaluation scripts (see eval/README.md)
├── tests/                             # Unit tests
├── arxiv_papers.db                    # Paper database, updated nightly by CI
├── pyproject.toml, uv.lock            # Dependencies (source of truth)
└── requirements.txt                   # Exported from uv.lock for Streamlit Cloud
```

## Configuration

Configuration is managed via:
- Environment variables (loaded via python-dotenv)
- Default values in `config.py`

Key configurable parameters:
- arXiv categories to follow
- Embedding model name
- Vector database path and settings
- LLM provider and model (HF API or local)
- Ingestion schedule
- Interface settings (port, theme, etc.)

### Hugging Face Inference Providers

The default model is `openai/gpt-oss-120b`, but it can only run if your Hugging Face account has enabled a compatible Inference Provider and your token can make Inference Providers calls. If the app reports that no provider supports the model, enable a compatible provider in your Hugging Face account settings or choose a model supported by a provider you have enabled. A model being available on Hugging Face does not guarantee that your account can call it through an Inference Provider.

To pin routing to a provider enabled for your account, set `HF_INFERENCE_PROVIDER` to its provider ID. If unset, Hugging Face selects automatically from providers enabled for the account. Keep `LLM_PROVIDER` as `huggingface_api`; it selects the application backend, not a Hugging Face inference provider.

## Dependencies

See `requirements.txt` for the full list. Key packages include:
- `sentence-transformers`
- `chromadb`
- `gradio`
- `requests`
- `python-dotenv`
- `huggingface_hub`
- `pydantic` (for config validation)

## Future Enhancements

- Full-text processing (PDF download and section-aware chunking)
- Hybrid search (BM25 + vector search)
- User personalization (saved searches, topic alerts)
- Advanced retrieval (re-ranking with cross-encoders)
- Multimodal support (figure/table extraction)
- Collaboration features (shared annotations, exports)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- arXiv.org for providing open access to scientific papers
- Hugging Face for open-source models and inference API
- The developers of sentence-transformers, ChromaDB, and Gradio

## Development (uv, tests, CI)

```bash
make lock                      # after changing pyproject.toml: updates uv.lock and requirements.txt (Streamlit Cloud)
uv sync --locked               # install exactly what is locked
uv run pytest -m "not slow and not network"   # unit tests + Streamlit AppTest (no network)
uv run streamlit run streamlit_app.py
```

* GitHub Actions: `ci.yml` (locked install, requirements/lock drift check, pytest incl. AppTest) and
  `nightly-ingest.yml` (daily arXiv ingestion, commits `arxiv_papers.db` when it changed).
* Groq calls retry on HTTP 429/5xx with backoff; if generation still fails the retrieved papers are shown.
* Answers must cite retrieved papers as `[n]`; uncited sentences are repaired once, then removed.
* `RETRIEVAL_MODE` = `dense` (default, unchanged behaviour) | `bm25` | `hybrid` | `hybrid_rerank`.
  Choose it from the numbers produced by the evaluation scripts, see [`eval/README.md`](eval/README.md).
