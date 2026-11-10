"""
Web server for arXiv Research Companion: a small JSON API plus the static front-end in web/.

Run locally:
    uvicorn arxiv_research_companion.server:app --reload

Deploy (Render, see render.yaml):
    uvicorn arxiv_research_companion.server:app --host 0.0.0.0 --port $PORT

The port opens immediately; the embedding model and the paper index are loaded in a
background thread, and /api/status reports ``ready`` once questions can be answered.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from arxiv_research_companion import app_state, rag_engine
from arxiv_research_companion.data_ingestion import (
    get_last_update_timestamp,
    get_total_papers_count,
)

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

MAX_QUESTION_CHARS = 500
DEFAULT_K = 3
MAX_K = 10
# The config default (100 tokens) is too short for a cited answer; used unless the env sets it.
DEFAULT_MAX_NEW_TOKENS = 800


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int = Field(default=DEFAULT_K, ge=1, le=MAX_K)


class RateLimiter:
    """Sliding one-minute window per client. In-memory: fine for a single process."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > 60.0:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            if len(self._hits) > 5000:  # bound memory if many distinct clients show up
                for key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[key]
            return True


class Runtime:
    """Everything the endpoints need, filled in by the background loader."""

    def __init__(self) -> None:
        self.ready = False
        self.error: Optional[str] = None
        self.config: Optional[Dict[str, Any]] = None
        self.vector_store: Optional[Dict[str, Any]] = None
        self.thread: Optional[threading.Thread] = None
        # One question at a time: the free instance has a fraction of a CPU and 512 MB.
        self.ask_lock = threading.Lock()

    def load(self) -> None:
        try:
            self.config, self.vector_store = app_state.build_state()
            self.ready = True
            logger.info("Index ready.")
        except Exception as exc:  # surfaced through /api/status instead of killing the server
            logger.exception("Could not build the paper index")
            self.error = f"{type(exc).__name__}: {exc}"

    def start(self) -> None:
        self.thread = threading.Thread(target=self.load, name="index-loader", daemon=True)
        self.thread.start()


def _client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")  # set by Render's proxy
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _date_text(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    return str(value)[:10] if value else ""


def _authors_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(a) for a in value]
    if isinstance(value, str) and value.strip():
        return [a.strip() for a in value.split(",") if a.strip()]
    return []


def _public_sources(sources: List[Dict[str, Any]], cited_ids: List[str]) -> List[Dict[str, Any]]:
    cited = set(cited_ids)
    out = []
    for n, source in enumerate(sources, 1):
        meta = source.get("metadata") or {}
        paper_id = str(source.get("id") or "")
        out.append({
            "n": n,
            "id": paper_id,
            "title": meta.get("title") or "Untitled",
            "authors": _authors_list(meta.get("authors")),
            "published": _date_text(meta.get("published")),
            "abstract": source.get("document") or "",
            "url": f"https://arxiv.org/abs/{paper_id}" if paper_id else "",
            "cited": paper_id in cited,
        })
    return out


def create_app() -> FastAPI:
    runtime = Runtime()
    limiter = RateLimiter(int(os.getenv("RATE_LIMIT_PER_MINUTE", "10")))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logging.basicConfig(level=logging.INFO)
        runtime.start()
        yield

    app = FastAPI(title="arXiv Research Companion", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.runtime = runtime

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    def status() -> Dict[str, Any]:
        if not runtime.ready:
            return {"ready": False, "error": runtime.error}
        config = runtime.config
        db_path = config["ingestion"]["db_path"]
        last = get_last_update_timestamp(db_path)
        llm = config["llm"]
        has_key = bool(llm["groq_api_key"] if llm["provider"] == "groq" else llm["api_token"])
        rcfg = config["retrieval"]
        return {
            "ready": True,
            "error": None,
            "papers": get_total_papers_count(db_path),
            "last_paper_date": _date_text(last),
            "model": llm["model_name"],
            "llm_configured": has_key,
            "retrieval_mode": rcfg.get("effective_mode", rcfg["mode"]),
        }

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request) -> Dict[str, Any]:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="The question is empty.")
        if not runtime.ready:
            raise HTTPException(status_code=503, detail=runtime.error or "The paper index is still loading.")
        if not limiter.allow(_client_id(request)):
            raise HTTPException(status_code=429, detail="Too many questions. Try again in a minute.")

        config = runtime.config
        llm = dict(config["llm"])
        if not os.getenv("LLM_MAX_NEW_TOKENS"):
            llm["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
        run_config = {**config, "llm": llm, "retrieval_k": body.k}

        with runtime.ask_lock:
            try:
                result = rag_engine.generate_answer(
                    query=question, vector_store=runtime.vector_store, config=run_config
                )
            except Exception:
                logger.exception("Error generating answer")
                # Last resort: never lose the sources just because generation blew up.
                try:
                    sources = rag_engine.retrieve_sources(question, runtime.vector_store, body.k)
                except Exception:
                    sources = []
                result = {
                    "answer": "Something went wrong while writing the answer. "
                              "The closest papers are listed below.",
                    "sources": sources,
                    "degraded": True,
                }

        citations = result.get("citations") or {}
        cited_ids = list(citations.get("cited_ids") or [])
        answer = result.get("answer") or ""
        if result.get("degraded"):
            # rag_engine appends a plain-text dump of the papers; the page lists them itself.
            answer = answer.split("\n\nHere are the papers I retrieved instead:")[0]
        return {
            "question": question,
            "answer": answer,
            "sources": _public_sources(result.get("sources") or [], cited_ids),
            "degraded": bool(result.get("degraded")),
            "refused": bool(result.get("refused")),
            "dropped_count": len(citations.get("dropped_sentences") or []),
        }

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


app = create_app()
