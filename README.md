# arXiv Research Companion

A research assistant that answers questions about recent arXiv papers (cs.AI, cs.LG, stat.ML...) with
Retrieval-Augmented Generation: it retrieves the most relevant abstracts, has an LLM write an answer
in which **every sentence cites the paper it comes from as `[n]`**, and refuses when the papers do not
contain the answer. The retrieval and the generation are both **measured** (see [Evaluation results](#evaluation-results)).

## Features

- Fetches papers from the arXiv API and keeps them in SQLite; incremental updates (only new papers) run nightly in CI
- Embeds abstracts with all-MiniLM-L6-v2 through ONNX Runtime (no PyTorch in production) and stores them in ChromaDB
- Four retrieval modes: dense, BM25, hybrid (reciprocal rank fusion) and hybrid + cross-encoder rerank; `dense` is the default, chosen by a rule declared before the measurement
- Answers with the LLM of your choice: local Ollama (no key), Groq, OpenAI, OpenRouter or Hugging Face. The page has a **Model** section to pick provider and model and to paste your own API key (sent with the question only, never stored on the server)
- Citation contract: sentences without a valid `[n]` are repaired once, then removed; if the papers do not answer the question the model must reply `NOT_ENOUGH_INFORMATION`, shown as a clean refusal
- Plain HTML/CSS/JS page served by a small FastAPI app; runs on CPU only; uses free APIs and open-source tools

## Architecture

1. **Data ingestion**: arXiv API, deduplication, SQLite (`arxiv_papers.db`)
2. **Embedding and vector storage**: ONNX MiniLM embeddings, ChromaDB similarity search
3. **RAG engine**: retrieval, prompt, LLM call with retries, citation enforcement, refusal detection
4. **Presentation**: FastAPI server and the static page in `web/`
5. **Configuration**: environment variables and defaults in `config.py`

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

## Evaluation results

Everything below is produced by `bash eval/run_all.sh` and `python -m eval.report` (full tables in
`results/RESULTS.md`, regenerated on every run and not tracked by git). Setup of the run reported
here: 300 abstracts (snapshot sha256 `002d5a0f...`), 40 hand-labelled questions (116 relevant papers),
14 out-of-corpus questions, generator `llama3.2` (3B) on Ollama, judge `qwen2.5:7b` (a different
family on purpose), retrieval `dense`, top-3 papers per question, temperature 0.7. All intervals
are 95 %. **One run is one sample**: the generator is stochastic, so generation figures move a little
from run to run.

### 1. Retrieval (no LLM involved)

| config | recall@3 | recall@10 | MRR@10 | nDCG@10 [95 % CI] |
|---|---|---|---|---|
| dense (MiniLM) | 0.388 | 0.600 | 0.617 | 0.516 [0.42, 0.61] |
| BM25 | 0.400 | 0.550 | 0.609 | 0.490 [0.39, 0.59] |
| hybrid (RRF) | 0.467 | 0.621 | 0.718 | **0.580** [0.48, 0.68] |
| hybrid + rerank | 0.383 | 0.633 | 0.655 | 0.542 [0.45, 0.63] |
| random (expected) | 0.011 | 0.035 | 0.029 | 0.022 |

* Every configuration is far above random. Hybrid has the best mean nDCG@10, but a paired bootstrap
  (Bonferroni-adjusted) shows it is **not significantly better** than dense (+0.064, CI [-0.018, +0.146])
  or hybrid + rerank. Only BM25 is significantly worse (+0.090, CI [+0.029, +0.152]).
* The pre-declared rule "take the cheapest configuration indistinguishable from the best" therefore
  selects **dense**, which is the default. The cross-encoder doubles the latency (p50 223 ms against
  110 ms) without a measurable gain.
* hit@k (a relevant paper among the top k) for dense: 0.47 / 0.70 / 0.80 / 0.88 at k = 1 / 3 / 5 / 10.
  The 3 papers given to the LLM contain a relevant one for 70 % of the questions.
* Questions where dense finds nothing in its top 10: q03, q05, q09, q15, q30 (5 of 40).

### 2. Cross-check on BEIR SciFact (5,183 docs, 300 queries, full test split)

| config | nDCG@10 [95 % CI] |
|---|---|
| dense | 0.645 [0.599, 0.689] |
| BM25 | 0.686 [0.642, 0.730] |
| hybrid | **0.717** [0.674, 0.758] |
| hybrid + rerank | 0.704 [0.660, 0.747] |

SciFact is claim verification over biomedical abstracts, not question answering over arXiv: it checks
that the embedding pipeline is sound, not the quality of this product's retrieval. The published
reference value for the model was not entered in `published_reference.json`, so no comparison with
the literature is claimed.

### 3. ONNX embeddings against sentence-transformers

On the 300 abstracts and the 40 questions the minimum cosine similarity between the two
implementations is 1.00000, the same top-1 neighbour is returned for 100 % of the questions and the
mean top-10 overlap is 1.000. The 0.99 threshold passes, so the PyTorch-free production embedder is
equivalent to the reference one. The SciFact nDCG@10 is identical (0.6451).

### 4. Generation (llama3.2, top-3 papers)

**Citations** (37 answered questions):

| measure | result |
|---|---|
| raw output already compliant (every factual sentence has a valid `[n]`) | 0.757 [0.599, 0.866] |
| factual sentences with a valid citation after enforcement | 1.000 [0.940, 1.000] (n = 60) |
| cited IDs that belong to the retrieved set | 1.000 [0.958, 1.000] (n = 87) |
| cited IDs that resolve on arXiv | 69 of 69 |

About a quarter of the raw answers contain uncited sentences (the only violation type observed), which
is why the repair-then-drop step exists. After it, no answer cites a paper that was not retrieved.

**Refusal** (`NOT_ENOUGH_INFORMATION`):

| measure | result |
|---|---|
| out-of-corpus questions refused | 14 of 14, 1.000 [0.785, 1.000] |
| in-scope questions refused by mistake | 3 of 40 = 0.075 [0.026, 0.199] (q01, q07, q13) |
| top-1 cosine AUROC, in-scope against out-of-corpus | 0.977 |

The top-1 cosine of in-scope questions ranges 0.37-0.66 (median 0.52) and that of out-of-corpus
questions 0.10-0.44 (median 0.26): a retrieval-score gate would separate them well (listed below as
a possible improvement). With 14 questions the lower bound of the refusal interval is wide.

**Is each sentence supported by its cited abstract?** Two judges, both scored against 30 sentence/abstract
pairs labelled by hand (12 supported, 12 partial, 6 unsupported; 24 real pairs and 6 deliberately
swapped ones):

| judge | accuracy | precision | recall | kappa |
|---|---|---|---|---|
| NLI cross-encoder `nli-deberta-v3-base` (threshold 0.20, leave-one-out) | 0.800 [0.627, 0.905] | 0.800 | 0.667 | 0.571 (AUC 0.824) |
| LLM judge `qwen2.5:7b` | 0.800 [0.627, 0.905] | 0.667 | 1.000 | 0.615 (3-class 0.570) |

Applied to the generated answers (calibrated judges):

| measure | NLI | LLM judge |
|---|---|---|
| (sentence, cited paper) pairs supported (n = 87) | 0.299 [0.213, 0.402] | 0.506 [0.403, 0.608] |
| sentences supported by at least one cited paper (n = 61) | 0.426 [0.310, 0.551] | 0.705 [0.581, 0.804] |

RAGAS faithfulness (judge `qwen2.5:7b`): **0.794** [0.684, 0.892] over 37 answers, no NaN.

Read these three numbers as a range, not as a fact: the NLI judge is strict (it recovers 67 % of
supported pairs), the LLM judge is generous (it accepts every supported pair but also 5 of the 12
"partial" and 1 of the 6 "unsupported" ones), and RAGAS uses the same LLM judge with its own prompt.
The truth is probably between 0.43 and 0.79.

### 5. Limits

* One labeller, no inter-annotator agreement; relevance was judged from arXiv.org, not pooled from
  system outputs, so relevant papers nobody labelled count as misses.
* 40 questions, 14 out-of-corpus questions and 30 calibration pairs: intervals are wide and most
  differences between configurations are not significant.
* The corpus is 300 recent abstracts, so recall@10 is easy; compare with the random row.
* Grounding is checked against the **abstract** only: a sentence true of the full paper but absent
  from the abstract counts as unsupported, and the "any cited paper supports it" rule under-counts
  claims that combine several papers.
* A 3B local model is a hard setting: it cites and refuses less reliably than a large hosted model, so
  generation figures are not comparable across providers. Retrieval figures are.
* The judges are imperfect proxies for a human reading, and the cited-sentence validator checks that a
  valid `[n]` is present, not that the sentence is meaningful.
* The nightly ingestion changes the live corpus; these numbers describe the frozen snapshot.

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
