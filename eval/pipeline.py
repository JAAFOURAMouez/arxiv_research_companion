"""Build the four retrieval configurations over a corpus and run queries through them."""
from __future__ import annotations

import tempfile
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from arxiv_research_companion.retrieval import (CrossEncoderReranker, DenseIndex, Retriever,
                                                build_retriever, doc_text)

CONFIGS = ("dense", "bm25", "hybrid", "hybrid_rerank")


class FakeReranker:
    """Token-overlap scorer.  SMOKE TESTS ONLY (never a real result)."""
    is_smoke = True

    def score(self, query, texts):
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


def make_reranker(kind: str, model_name: Optional[str] = None):
    if kind == "ce":
        return CrossEncoderReranker(model_name) if model_name else CrossEncoderReranker()
    if kind == "fake":
        return FakeReranker()
    raise ValueError(f"unknown reranker {kind!r}")


def build_retrievers(papers: Sequence[dict], doc_vectors: np.ndarray, embedder,
                     configs: Sequence[str], *, reranker=None, doc_text_mode: str = "abstract",
                     dense_backend: str = "numpy", candidate_depth: int = 50, rerank_depth: int = 20,
                     rrf_k: int = 60) -> Dict[str, Retriever]:
    ids = [p["arxiv_id"] for p in papers]
    kw = dict(embedder=embedder, papers=papers, doc_text_mode=doc_text_mode,
              candidate_depth=candidate_depth, rerank_depth=rerank_depth, rrf_k=rrf_k)
    if dense_backend == "numpy":
        kw["dense_index"] = DenseIndex(ids, doc_vectors)
    elif dense_backend == "chroma":
        import chromadb
        client = chromadb.PersistentClient(path=tempfile.mkdtemp(prefix="eval_chroma_"))
        coll = client.create_collection("eval", metadata={"hnsw:space": "cosine"})
        for s in range(0, len(ids), 500):
            coll.add(ids=ids[s:s + 500], embeddings=doc_vectors[s:s + 500].tolist())
        kw["collection"] = coll
    else:
        raise ValueError(dense_backend)
    out = {}
    for c in configs:
        if c == "hybrid_rerank":
            if reranker is None:
                raise ValueError("hybrid_rerank needs a reranker")
            out[c] = build_retriever(c, reranker=reranker, **kw)
        else:
            out[c] = build_retriever(c, **kw)
    return out


def run_queries(retriever: Retriever, queries: Sequence[dict], depth: int):
    """-> (rankings {qid: [doc_id...]}, latency_ms list).  Latency includes query embedding."""
    rankings, lat = {}, []
    for q in queries:
        t0 = time.perf_counter()
        hits = retriever.retrieve(q["question"], depth)
        lat.append((time.perf_counter() - t0) * 1000.0)
        rankings[q["qid"]] = [h.doc_id for h in hits]
    return rankings, lat


def latency_summary(lat_ms: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(lat_ms, dtype=float)
    return {"p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2),
            "mean_ms": round(float(a.mean()), 2)}
