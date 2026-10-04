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

- Python 3.8+
- Git
- (Optional) GPU for faster embeddings (CPU-only works fine)

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:JAAFOURAMouez/arxiv_research_companion.git
   cd arxiv_research_companion
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to add your Hugging Face API token if desired
   ```

### Usage

To start the application (which will also ingest initial data):

```bash
python -m arxiv_research_companion.main
```

Then open your browser to the URL shown in the output (typically http://localhost:7860).

## Project Structure

```
arxiv_research_companion/
├── arxiv_research_companion/          # Main package
│   ├── __init__.py
│   ├── data_ingestion.py              # arXiv API ingestion
│   ├── embedding_store.py             # Embedding generation & vector storage
│   ├── rag_engine.py                  # RAG pipeline: retrieve + generate
│   ├── interface.py                   # Gradio web interface
│   ├── config.py                      # Configuration management
│   └── utils.py                       # Helper functions
├── tests/                             # Unit tests
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

## Configuration

Configuration is managed via:
- Environment variables (loaded via python-dotenv)
- A `config.yaml` file (optional, for non-secrets)
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