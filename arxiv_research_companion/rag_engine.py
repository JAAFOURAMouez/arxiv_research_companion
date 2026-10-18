"""
Retrieval and Generation Layer (RAG Engine) for arXiv Research Companion.

Responsibilities:
- Retrieve the top-k paper abstracts for a query (via ``vector_store["retriever"]`` when
  present - dense / bm25 / hybrid / hybrid+rerank - otherwise the legacy Chroma query).
- Format the abstracts as a numbered context ``[1]..[k]`` and prompt the LLM.
- Optionally enforce the citation contract (see ``citations.py``): validate, ask the model
  to repair once, then drop factual sentences that still carry no valid ``[n]``.
- Let the model refuse (sentinel ``NOT_ENOUGH_INFORMATION``) when the context is insufficient.
- Retry Groq calls on HTTP 429/5xx with bounded backoff (``llm_http.py``).
- ALWAYS return the retrieved sources, even when the LLM fails: the sources are built
  right after retrieval, before any LLM call.
"""

import logging
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from arxiv_research_companion import citations
from arxiv_research_companion.llm_http import GROQ_CHAT_URL, LlmHttpError, chat_completion  # noqa: F401

logger = logging.getLogger(__name__)

NO_PAPERS_MESSAGE = (
    "I couldn't find any papers to answer this question. Try again after papers have been ingested."
)
REFUSAL_MESSAGE = (
    "The retrieved papers don't contain enough information to answer this question. "
    "The closest papers I found are listed below."
)
UNCITED_MESSAGE = (
    "I could not produce an answer whose claims are all cited to the retrieved papers, so I am "
    "not showing one. The retrieved papers are listed below."
)


# --------------------------------------------------------------------------- #
# Retrieval                                                                   #
# --------------------------------------------------------------------------- #
def _hydrate(ids: List[str], vector_store: Dict[str, Any]):
    """Look up (metadata, document) for each id from papers_by_id or the collection."""
    by_id = vector_store.get("papers_by_id")
    metadatas, documents = [], []
    missing = [i for i in ids if not by_id or i not in by_id]
    fetched: Dict[str, Any] = {}
    if missing and vector_store.get("collection") is not None:
        got = vector_store["collection"].get(ids=missing, include=["metadatas", "documents"])
        for i, meta, doc in zip(got.get("ids", []), got.get("metadatas", []), got.get("documents", [])):
            fetched[i] = (meta or {}, doc or "")
    for i in ids:
        if by_id and i in by_id:
            paper = by_id[i]
            metadatas.append({k: v for k, v in paper.items() if k != "summary"})
            documents.append(paper.get("summary") or "")
        else:
            meta, doc = fetched.get(i, ({}, ""))
            metadatas.append(meta)
            documents.append(doc)
    return metadatas, documents


def _retrieve(query: str, vector_store: Dict[str, Any], retrieval_k: int) -> Dict[str, Any]:
    """Return Chroma-shaped results: ids/distances/metadatas/documents (one outer list)."""
    retriever = vector_store.get("retriever")
    if retriever is not None:
        hits = retriever.retrieve(query, retrieval_k)
        ids = [h.doc_id for h in hits]
        metadatas, documents = _hydrate(ids, vector_store)
        return {
            "ids": [ids],
            "distances": [[1.0 - h.score for h in hits]],
            "metadatas": [metadatas],
            "documents": [documents],
        }

    model = vector_store["model"]
    query_embedding = model.encode([query])
    results = vector_store["collection"].query(
        query_embeddings=query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding,
        n_results=retrieval_k,
        include=["metadatas", "documents"],
    )
    ids_list = results.get("ids", [[]])[0] if results.get("ids") else []

    def first_query_values(key):
        values = results.get(key) or []
        if not values:
            return []
        # Chroma returns one list per query. Some legacy mocks modeled one
        # singleton list per result, so flatten that shape for compatibility.
        if len(values) == len(ids_list) and all(isinstance(value, list) for value in values):
            return [value[0] if value else None for value in values]
        return values[0] if isinstance(values[0], list) else values

    return {
        "ids": [ids_list],
        "distances": [results.get("distances", [[]])[0]] if results.get("distances") else [[]],
        "metadatas": [first_query_values("metadatas")],
        "documents": [first_query_values("documents")],
    }


def _build_sources(formatted_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    try:
        ids_list = formatted_results["ids"][0] if formatted_results.get("ids") else []
        metadatas = formatted_results["metadatas"][0] if formatted_results.get("metadatas") else []
        documents = formatted_results["documents"][0] if formatted_results.get("documents") else []
        for i, paper_id in enumerate(ids_list):
            sources.append({
                "id": paper_id,
                "metadata": (metadatas[i] if i < len(metadatas) else {}) or {},
                "document": (documents[i] if i < len(documents) else "") or "",
            })
    except Exception as e:  # defensive: never let source formatting hide the answer
        logger.warning("Error formatting sources: %s", e)
        return []
    return sources


def retrieve_sources(query: str, vector_store: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """Retrieval only (no LLM).  Used as a last-resort fallback by the UI."""
    return _build_sources(_retrieve(query, vector_store, k))


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #
def generate_answer(query: str, vector_store: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an answer to a user query using the RAG pipeline.

    Config (nested ``llm`` / ``citations`` sections; legacy flat LLM keys still accepted):
        llm.model_name, max_new_tokens, temperature, top_p, timeout, provider,
        api_token, groq_api_key, inference_provider, max_retries, retry_base_delay
        citations.enforce (bool, default False here; the app config turns it on),
        citations.policy ("drop" | "flag"), citations.max_repair_attempts (default 1)
        retrieval_k (top level)

    Returns a dict with:
        answer, sources (always the retrieved papers), degraded (LLM failed),
        llm_error (str|None), refused (bool), raw_answer, citations (dict|None).
    """
    if not query:
        logger.warning("Empty query provided to generate_answer")
        return {"answer": "Please provide a question.", "sources": []}

    llm_config = config.get("llm") or {}
    llm_model = llm_config.get("model_name", config.get("llm_model", "openai/gpt-oss-120b"))
    llm_max_new_tokens = llm_config.get("max_new_tokens", config.get("llm_max_new_tokens", 100))
    llm_temperature = llm_config.get("temperature", config.get("llm_temperature", 0.7))
    llm_top_p = llm_config.get("top_p", config.get("llm_top_p"))
    hf_inference_provider = llm_config.get("inference_provider") or config.get("hf_inference_provider") or None
    retrieval_k = config.get("retrieval_k", 2)
    hf_api_token = llm_config.get("api_token") or config.get("hf_api_token") or None
    llm_timeout = llm_config.get("timeout", config.get("llm_timeout", 60.0))
    llm_provider = (llm_config.get("provider") or "huggingface_api").lower()
    groq_api_key = llm_config.get("groq_api_key") or None
    max_retries = int(llm_config.get("max_retries", 3))
    retry_base_delay = float(llm_config.get("retry_base_delay", 1.0))

    cit_cfg = config.get("citations") or {}
    enforce_citations = bool(cit_cfg.get("enforce", False))
    citation_policy = cit_cfg.get("policy", "drop")
    max_repairs = int(cit_cfg.get("max_repair_attempts", 1))

    # Step 1: retrieval.  Sources are fixed here, before the LLM is ever called.
    try:
        logger.info("Retrieving top %s papers for query: '%s...'", retrieval_k, query[:50])
        formatted_results = _retrieve(query, vector_store, retrieval_k)
    except Exception as e:
        logger.error("Error retrieving papers from vector store: %s", e)
        return {"answer": f"Error retrieving information: {str(e)}", "sources": []}

    context = _format_context(formatted_results)
    if context == "No relevant papers found.":
        return {"answer": NO_PAPERS_MESSAGE, "sources": []}
    sources = _build_sources(formatted_results)
    source_ids = [s["id"] for s in sources]

    prompt = (
        _construct_cited_prompt(query, context, len(source_ids))
        if enforce_citations
        else _construct_prompt(query, context)
    )

    def complete(messages: List[Dict[str, str]]) -> str:
        if llm_provider == "groq":
            return _call_groq_messages(
                messages=messages, model=llm_model, api_key=groq_api_key,
                max_tokens=llm_max_new_tokens, temperature=llm_temperature,
                top_p=llm_top_p, timeout=llm_timeout,
                max_attempts=max_retries + 1, base_delay=retry_base_delay,
            )
        client_options = {"model": llm_model, "token": hf_api_token, "timeout": llm_timeout}
        if hf_inference_provider:
            client_options["provider"] = hf_inference_provider
        client = InferenceClient(**client_options)
        generation_options = {
            "messages": messages,
            "max_tokens": llm_max_new_tokens,
            "temperature": llm_temperature,
        }
        if llm_top_p is not None:
            generation_options["top_p"] = llm_top_p
        response = client.chat_completion(**generation_options)
        choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
        if not choices:
            raise ValueError("Chat completion returned no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
        return message.get("content") if isinstance(message, dict) else getattr(message, "content", None)

    # Step 2: first LLM call.
    messages = [{"role": "user", "content": prompt}]
    try:
        content = complete(messages)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("The model returned an empty answer")
        content = content.strip()
    except Exception as e:
        logger.exception("Error generating answer with LLM for model %s", llm_model)
        return {
            "answer": _degraded_answer(llm_model, e, context),
            "sources": sources,
            "degraded": True,
            "llm_error": f"{type(e).__name__}: {e}",
            "refused": False,
            "citations": None,
        }

    result: Dict[str, Any] = {
        "sources": sources, "degraded": False, "llm_error": None, "refused": False, "citations": None,
    }

    if not enforce_citations:
        result["answer"] = content
        return result

    # Step 3: citation enforcement (validate -> repair once -> drop uncited sentences).
    result["raw_answer"] = content
    report = citations.validate_answer(content, source_ids)
    first_pass_ok = report.ok
    repairs_used = 0
    while not report.ok and not report.refused and repairs_used < max_repairs:
        messages = messages + [
            {"role": "assistant", "content": content},
            {"role": "user", "content": citations.correction_message(report)},
        ]
        try:
            repaired = complete(messages)
        except Exception as e:  # keep the first answer; it will be enforced below
            logger.warning("Citation repair call failed: %s", e)
            break
        if not isinstance(repaired, str) or not repaired.strip():
            break
        repairs_used += 1
        content = repaired.strip()
        report = citations.validate_answer(content, source_ids)

    info: Dict[str, Any] = {
        "first_pass_ok": first_pass_ok,
        "repair_attempts": repairs_used,
        "ok_after_repair": report.ok,
        **report.summary(),
    }
    if report.refused:
        result.update(answer=REFUSAL_MESSAGE, refused=True, citations=info)
        return result

    answer, enforcement = citations.enforce(content, source_ids, policy=citation_policy)
    info.update(enforcement)
    if not answer:
        answer = UNCITED_MESSAGE
        result["degraded"] = True
        result["llm_error"] = "no sentence carried a valid citation"
    result["answer"] = answer
    result["citations"] = info
    return result


def _degraded_answer(model: str, error: Exception, context: str) -> str:
    """Message shown when generation fails: says why, then lists the retrieved papers."""
    text = str(error)
    if "model_not_supported" in text or "not supported by any provider" in text:
        hint = (
            "Enable an Inference Provider for your account (or choose a model supported by a "
            "provider you have enabled) and ensure your token can call Inference Providers."
        )
    elif "no choices" in text:
        hint = (
            "The response contained no choices; a compatible Hugging Face chat inference "
            "provider may be unavailable for this model."
        )
    elif isinstance(error, LlmHttpError) and error.status == 429:
        hint = f"The provider is rate-limiting requests (HTTP 429, {error.attempts} attempts). Try again shortly."
    elif isinstance(error, LlmHttpError) and error.status and error.status >= 500:
        hint = f"The provider returned HTTP {error.status} after {error.attempts} attempts. Try again shortly."
    elif isinstance(error, ValueError) and "GROQ_API_KEY" in text:
        hint = "No Groq API key is configured."
    elif "empty answer" in text:
        hint = "The model returned an empty answer."
    else:
        hint = f"Error: {text[:200]}"
    return (
        f"I failed to generate an answer with model '{model}'. {hint}\n\n"
        f"Here are the papers I retrieved instead:\n\n{context}"
    )


# --------------------------------------------------------------------------- #
# Groq                                                                        #
# --------------------------------------------------------------------------- #
def _call_groq_messages(
    messages: List[Dict[str, str]],
    model: str,
    api_key: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    timeout: float,
    max_attempts: int = 4,
    base_delay: float = 1.0,
) -> str:
    return chat_completion(
        messages, model=model, api_key=api_key, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, timeout=timeout,
        max_attempts=max_attempts, base_delay=base_delay,
    )


def _call_groq(
    prompt: str,
    model: str,
    api_key: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    timeout: float,
) -> str:
    """Single-turn convenience wrapper kept for backward compatibility."""
    return _call_groq_messages(
        [{"role": "user", "content": prompt}], model, api_key, max_tokens, temperature, top_p, timeout
    )


# --------------------------------------------------------------------------- #
# Prompt construction                                                         #
# --------------------------------------------------------------------------- #
def _format_context(retrieval_results: Dict[str, Any]) -> str:
    """Numbered context ``[n] arXiv:ID`` + title, authors, abstract for each paper."""
    if not retrieval_results or not retrieval_results.get("ids") or not retrieval_results["ids"][0]:
        return "No relevant papers found."

    context_parts = []
    ids_list = retrieval_results["ids"][0]
    metadatas_list = retrieval_results["metadatas"][0]
    documents_list = retrieval_results["documents"][0]

    for i, paper_id in enumerate(ids_list):
        metadata = (metadatas_list[i] if i < len(metadatas_list) else {}) or {}
        document = documents_list[i] if i < len(documents_list) else ""
        title = metadata.get("title", "Untitled")
        authors = metadata.get("authors", [])
        authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        context_parts.append(
            f"[{i + 1}] arXiv:{paper_id}\nTitle: {title}\nAuthors: {authors_str}\nAbstract: {document}\n"
        )
    return "\n---\n".join(context_parts)


def _construct_prompt(query: str, context: str) -> str:
    """Legacy prompt (no citation contract)."""
    if not context or context == "No relevant papers found.":
        return f"""Answer the question based on the context.

Context: {context}

Question: {query}

Answer:"""
    return f"""Answer the question based on the context. Use only the information from the context to formulate your answer. If the context does not contain enough information to answer the question, say that you cannot answer based on the provided information.

Context:
{context}

Question:
{query}

Answer:"""


def _construct_cited_prompt(query: str, context: str, k: int) -> str:
    """Prompt that imposes the ``[n]`` citation contract and the refusal sentinel."""
    return f"""Answer the question based on the context. Use only the information from the context to formulate your answer.

Rules:
1. The context holds {k} numbered papers, [1] to [{k}]. End every sentence that states a fact with the number(s) of the supporting paper(s), placed before the final period, for example: "Method X improves Y [2]." For several papers use separate brackets: [1][3].
2. Only use numbers from 1 to {k}. Never write arXiv IDs or titles as citations.
3. Do not add any information that is not in the context.
4. If the context does not contain enough information to answer the question, reply with exactly {citations.REFUSAL_SENTINEL} and nothing else.

Context:
{context}

Question:
{query}

Answer:"""
