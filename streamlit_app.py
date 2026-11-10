"""
Streamlit front-end for arXiv Research Companion.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud:
    Main file path: streamlit_app.py
    Secrets (Settings -> Secrets): see secrets.toml.example
"""

import logging
import os

import streamlit as st

# --------------------------------------------------------------------------- #
# Secrets -> environment variables (must happen BEFORE load_config is called) #
# --------------------------------------------------------------------------- #
_SECRET_KEYS = (
    "GROQ_API_KEY",
    "LLM_PROVIDER",
    "HF_API_TOKEN",
    "HF_INFERENCE_PROVIDER",
    "LLM_MODEL_NAME",
    "LLM_MAX_NEW_TOKENS",
    "LLM_TIMEOUT",
    "LLM_MAX_RETRIES",
    "EMBEDDING_MODEL_NAME",
    "RETRIEVAL_MODE",
    "CITATIONS_ENFORCE",
    "CITATIONS_POLICY",
)


def _export_secrets_to_env() -> None:
    """Copy Streamlit secrets (if any) into os.environ so config.load_config() sees them."""
    try:
        for key in _SECRET_KEYS:
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        # No secrets.toml (e.g. local run with a .env file): nothing to do.
        pass


_export_secrets_to_env()

from arxiv_research_companion.app_state import build_state, refresh_retriever  # noqa: E402
from arxiv_research_companion.data_ingestion import (  # noqa: E402
    get_last_update_timestamp,
    get_stored_papers,
    get_total_papers_count,
    ingest_new_papers,
)
from arxiv_research_companion.embedding_store import index_papers  # noqa: E402
from arxiv_research_companion.rag_engine import generate_answer, retrieve_sources  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXAMPLES = [
    "What are the latest developments in transformer architectures?",
    "How does reinforcement learning apply to robotics?",
    "What are recent advances in diffusion models for image generation?",
    "What are the challenges in scaling large language models?",
]

st.set_page_config(page_title="arXiv Research Companion", page_icon="📚", layout="wide")

# --------------------------------------------------------------------------- #
# Cached resources                                                            #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model and indexing papers...")
def get_app_state():
    """Built once per process; see app_state.build_state()."""
    return build_state()


config, vector_store = get_app_state()
db_path = config["ingestion"]["db_path"]

# --------------------------------------------------------------------------- #
# Sidebar                                                                     #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of papers retrieved (k)", 1, 10, 3)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max answer tokens", 100, 2000, 800, 50)

    st.divider()
    st.subheader("Database")
    try:
        last_update = get_last_update_timestamp(db_path)
        total = get_total_papers_count(db_path)
        st.metric("Papers in database", total)
        st.caption(
            "Last paper date: "
            + (last_update.strftime("%Y-%m-%d %H:%M") if last_update else "never")
        )
        st.caption(f"Papers indexed for search: {vector_store['collection'].count()}")
    except Exception as exc:  # pragma: no cover - defensive UI code
        st.warning(f"Could not read database status: {exc}")

    rcfg = config["retrieval"]
    st.caption(f"Retrieval: {rcfg.get('effective_mode', rcfg['mode'])}")
    if rcfg.get("fallback_reason"):
        st.warning(f"Retrieval fallback: {rcfg['fallback_reason']}")

    if st.button("Fetch new arXiv papers", use_container_width=True):
        with st.spinner("Fetching from arXiv and indexing..."):
            try:
                categories = config["arxiv"]["categories"]
                ingest_new_papers(
                    categories,
                    db_path,
                    max_results_per_request=config["arxiv"]["max_results_per_request"],
                    max_total=config["arxiv"]["max_total_per_run"],
                )
                papers = get_stored_papers(db_path)
                indexed = index_papers(vector_store, papers)
                refresh_retriever(config, vector_store, papers)
                st.success(f"Done. {indexed} new papers indexed.")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")
        st.caption("Note: on Streamlit Cloud, new papers are lost when the app restarts.")

    provider = config["llm"]["provider"]
    st.caption(f"LLM: {config['llm']['model_name']} via {provider}")
    if provider == "groq" and not config["llm"]["groq_api_key"]:
        st.warning("No Groq key found. Add `GROQ_API_KEY` in the app secrets.")
    elif provider != "groq" and not config["llm"]["api_token"]:
        st.warning(
            "No LLM key found. Add `GROQ_API_KEY` (or `HF_API_TOKEN`) in the app secrets "
            "(or in a local `.env`) to enable answer generation."
        )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Main chat                                                                   #
# --------------------------------------------------------------------------- #
st.title("📚 arXiv Research Companion")
st.write("Ask questions about recent AI/ML research papers from arXiv.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _authors_text(authors):
    return ", ".join(authors) if isinstance(authors, (list, tuple)) else str(authors)


def render_sources(sources, cited_ids=None):
    """Show the retrieved papers as expandable cards with arXiv links."""
    if not sources:
        return
    cited = set(cited_ids or [])
    st.markdown("**Sources**")
    for i, source in enumerate(sources, 1):
        meta = source.get("metadata") or {}
        title = meta.get("title", "Untitled")
        paper_id = source.get("id", "")
        tag = " ✔ cited" if paper_id in cited else ""
        with st.expander(f"[{i}] {title}{tag}"):
            if meta.get("authors"):
                st.caption(f"Authors: {_authors_text(meta['authors'])}")
            if meta.get("published"):
                st.caption(f"Published: {meta['published']}")
            st.write(source.get("document", ""))
            if paper_id:
                st.markdown(f"[View on arXiv](https://arxiv.org/abs/{paper_id})")


def render_notes(result):
    """Surface degraded / refused / citation-enforcement state to the reader."""
    if not result:
        return
    if result.get("degraded"):
        st.warning("The answer could not be generated; the retrieved papers are shown below.")
    cit = result.get("citations") or {}
    dropped = cit.get("dropped_sentences") or []
    if dropped:
        st.caption(
            f"{len(dropped)} sentence(s) without a valid citation were removed from this answer."
        )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_notes(message.get("result"))
        render_sources(message.get("sources"), message.get("cited_ids"))

if not st.session_state.messages:
    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, key=f"ex_{example}"):
            st.session_state.pending_question = example
            st.rerun()

question = st.chat_input("Ask about recent AI/ML research...")
if not question:
    question = st.session_state.pop("pending_question", None)

if question and question.strip():
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating an answer..."):
            run_config = {
                **config,
                "llm": {
                    **config["llm"],
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                },
                "retrieval_k": k,
            }
            result = None
            try:
                result = generate_answer(
                    query=question.strip(),
                    vector_store=vector_store,
                    config=run_config,
                )
                answer, sources = result["answer"], result["sources"]
            except Exception as exc:
                logger.exception("Error generating answer")
                # Last resort: never lose the sources just because generation blew up.
                try:
                    sources = retrieve_sources(question.strip(), vector_store, k)
                except Exception:
                    sources = []
                answer = f"Sorry, I encountered an error: {exc}"
                result = {"degraded": True}

        st.markdown(answer)
        render_notes(result)
        cited_ids = ((result or {}).get("citations") or {}).get("cited_ids")
        render_sources(sources, cited_ids)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "cited_ids": cited_ids,
            "result": {
                "degraded": (result or {}).get("degraded"),
                "citations": (result or {}).get("citations"),
            },
        }
    )
