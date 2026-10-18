"""
Retrieval strategies shared by the Streamlit app and the evaluation scripts.

The evaluation (``eval/``) imports *this* module, so what gets measured is the code
that ships.  Four modes are supported:

``dense``          cosine similarity over MiniLM embeddings
``bm25``           SQLite FTS5 BM25 (OR-query over stop-word-filtered tokens)
``hybrid``         Reciprocal Rank Fusion (RRF) of ``dense`` and ``bm25``
``hybrid_rerank``  ``hybrid`` candidates re-scored by a cross-encoder

Nothing here imports torch / sentence-transformers / chromadb at module import time.
The cross-encoder (the only torch consumer) is imported lazily and only when
``hybrid_rerank`` is requested.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_DEPTH = 50
DEFAULT_RERANK_DEPTH = 20
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Deliberately small: BM25's IDF already down-weights frequent words.  The list is part
# of the "bm25" configuration being evaluated, so it is documented rather than tuned.
_STOPWORDS = frozenset(
    """a an the and or but if of in on at to for from by with without about as into over
    under between is are was were be been being do does did have has had can could should
    would may might will shall what which who whom whose when where why how this that these
    those it its their there than then so such not no nor too very we our us you your i me
    my he she they them his her paper papers""".split()
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# --------------------------------------------------------------------------- #
# Document text                                                               #
# --------------------------------------------------------------------------- #
def doc_text(paper: Dict, mode: str = "abstract") -> str:
    """Text indexed for a paper.  ``abstract`` mirrors what the app embeds today."""
    summary = (paper.get("summary") or "").strip()
    if mode == "abstract":
        return summary
    if mode == "title_abstract":
        title = (paper.get("title") or "").strip()
        return f"{title}. {summary}" if title else summary
    raise ValueError(f"unknown doc_text mode: {mode!r}")


# --------------------------------------------------------------------------- #
# BM25 via SQLite FTS5                                                        #
# --------------------------------------------------------------------------- #
def fts5_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.close()
        return True
    except sqlite3.Error:
        return False


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query ('' when nothing is left)."""
    seen: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOPWORDS or tok in seen:
            continue
        seen.append(tok)
    return " OR ".join(f'"{t}"' for t in seen)


class Bm25Index:
    """In-memory FTS5 index (``porter unicode61`` tokenizer, SQLite's ``bm25()``)."""

    def __init__(self, docs: Sequence[Tuple[str, str]]):
        """``docs`` is a sequence of ``(doc_id, text)``."""
        if not fts5_available():
            raise RuntimeError("This SQLite build has no FTS5 support")
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute(
            "CREATE VIRTUAL TABLE fts USING fts5("
            "doc_id UNINDEXED, body, tokenize='porter unicode61')"
        )
        self._conn.executemany(
            "INSERT INTO fts(doc_id, body) VALUES (?, ?)", list(docs)
        )
        self._conn.commit()
        self.size = len(docs)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Top-``k`` ``(doc_id, score)``; score = ``-bm25`` so that larger is better."""
        match = fts_query(query)
        if not match or k <= 0:
            return []
        rows = self._conn.execute(
            "SELECT doc_id, bm25(fts, 0.0, 1.0) AS s FROM fts "
            "WHERE fts MATCH ? ORDER BY s ASC, doc_id ASC LIMIT ?",
            (match, int(k)),
        ).fetchall()
        return [(doc_id, -float(s)) for doc_id, s in rows]

    def close(self) -> None:
        self._conn.close()


# --------------------------------------------------------------------------- #
# Dense search                                                                #
# --------------------------------------------------------------------------- #
def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class DenseIndex:
    """Exact (brute-force) cosine search.  Deterministic tie-break on doc_id."""

    def __init__(self, ids: Sequence[str], matrix: np.ndarray):
        if len(ids) != len(matrix):
            raise ValueError("ids and matrix have different lengths")
        self.ids = np.array(list(ids), dtype=object)
        self.matrix = _normalize(matrix)

    def search_vector(self, vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        q = _normalize(vector)[0]
        scores = self.matrix @ q
        order = np.lexsort((self.ids.astype(str), -scores))[:k]
        return [(str(self.ids[i]), float(scores[i])) for i in order]


DenseSearch = Callable[[str, int], List[Tuple[str, float]]]


def make_numpy_dense_search(embedder, index: DenseIndex) -> DenseSearch:
    def search(query: str, k: int) -> List[Tuple[str, float]]:
        vec = np.asarray(embedder.encode([query]))
        return index.search_vector(vec, k)

    return search


def make_chroma_dense_search(embedder, collection) -> DenseSearch:
    """Dense search through a Chroma collection (what the Streamlit app used so far)."""

    def search(query: str, k: int) -> List[Tuple[str, float]]:
        vec = np.asarray(embedder.encode([query]), dtype=np.float32)
        n = min(int(k), int(collection.count()))  # some Chroma versions raise if k > count
        if n <= 0:
            return []
        res = collection.query(
            query_embeddings=vec.tolist(), n_results=n, include=["distances"]
        )
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        # Chroma's "cosine" space returns 1 - cos.  Convert back to a similarity.
        return [(i, 1.0 - float(d)) for i, d in zip(ids, dists)]

    return search


# --------------------------------------------------------------------------- #
# Fusion and reranking                                                        #
# --------------------------------------------------------------------------- #
def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion.  Ties are broken on doc_id for determinism."""
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    scores: Dict[str, float] = {}
    for weight, ranking in zip(weights, rankings):
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class CrossEncoderReranker:
    """Thin wrapper around ``sentence_transformers.CrossEncoder`` (lazy import)."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        from sentence_transformers import CrossEncoder  # heavy: torch

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        out = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(x) for x in out]


# --------------------------------------------------------------------------- #
# Retriever                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    doc_id: str
    score: float
    rank: int  # 1-based


class Retriever:
    """Single entry point: ``retrieve(query, k)`` -> ranked ``Hit`` list."""

    def __init__(
        self,
        mode: str,
        *,
        dense_search: Optional[DenseSearch] = None,
        bm25: Optional[Bm25Index] = None,
        reranker=None,
        text_of: Optional[Callable[[str], str]] = None,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        rerank_depth: int = DEFAULT_RERANK_DEPTH,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        needs_dense = mode in ("dense", "hybrid", "hybrid_rerank")
        needs_bm25 = mode in ("bm25", "hybrid", "hybrid_rerank")
        if needs_dense and dense_search is None:
            raise ValueError(f"mode {mode!r} needs dense_search")
        if needs_bm25 and bm25 is None:
            raise ValueError(f"mode {mode!r} needs a Bm25Index")
        if mode == "hybrid_rerank" and (reranker is None or text_of is None):
            raise ValueError("mode 'hybrid_rerank' needs reranker and text_of")
        self.mode = mode
        self.dense_search = dense_search
        self.bm25 = bm25
        self.reranker = reranker
        self.text_of = text_of
        self.candidate_depth = candidate_depth
        self.rerank_depth = rerank_depth
        self.rrf_k = rrf_k

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _hits(pairs: Sequence[Tuple[str, float]], k: int) -> List[Hit]:
        return [Hit(d, float(s), r) for r, (d, s) in enumerate(pairs[:k], start=1)]

    def dense_top1_score(self, query: str) -> Optional[float]:
        """Cosine of the best dense match (used by the refusal analysis)."""
        if self.dense_search is None:
            return None
        res = self.dense_search(query, 1)
        return res[0][1] if res else None

    def _fused(self, query: str, depth: int) -> List[Tuple[str, float]]:
        depth = max(depth, self.candidate_depth)
        dense = [d for d, _ in self.dense_search(query, depth)]
        sparse = [d for d, _ in self.bm25.search(query, depth)]
        return rrf_fuse([dense, sparse], k=self.rrf_k)

    # -- API -------------------------------------------------------------- #
    def retrieve(self, query: str, k: int) -> List[Hit]:
        if not query or not query.strip() or k <= 0:
            return []
        if self.mode == "dense":
            return self._hits(self.dense_search(query, k), k)
        if self.mode == "bm25":
            return self._hits(self.bm25.search(query, k), k)
        fused = self._fused(query, max(k, self.rerank_depth))
        if self.mode == "hybrid":
            return self._hits(fused, k)
        # hybrid_rerank: re-score the top `rerank_depth` fused candidates; anything
        # deeper keeps its RRF order *after* them so deep rankings (MRR, nDCG) stay valid.
        head = fused[: self.rerank_depth]
        tail = fused[self.rerank_depth :]
        scores = self.reranker.score(query, [self.text_of(d) for d, _ in head])
        reranked = sorted(
            zip((d for d, _ in head), scores), key=lambda x: (-x[1], x[0])
        )
        floor = (reranked[-1][1] if reranked else 0.0) - 1.0
        tail_scored = [(d, floor - i * 1e-6) for i, (d, _) in enumerate(tail)]
        return self._hits(list(reranked) + tail_scored, k)


def build_retriever(
    mode: str,
    *,
    embedder,
    papers: Sequence[Dict],
    collection=None,
    dense_index: Optional[DenseIndex] = None,
    reranker=None,
    doc_text_mode: str = "abstract",
    **kwargs,
) -> Retriever:
    """Assemble a ``Retriever`` from stored papers.

    Dense search uses ``collection`` (Chroma) when given, otherwise ``dense_index``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    by_id = {p["arxiv_id"]: p for p in papers}
    dense_search = None
    if mode != "bm25":
        if collection is not None:
            dense_search = make_chroma_dense_search(embedder, collection)
        elif dense_index is not None:
            dense_search = make_numpy_dense_search(embedder, dense_index)
        else:
            raise ValueError("dense modes need `collection` or `dense_index`")
    bm25 = None
    if mode != "dense":
        bm25 = Bm25Index([(p["arxiv_id"], doc_text(p, doc_text_mode)) for p in papers])
    text_of = lambda d: doc_text(by_id[d], doc_text_mode)  # noqa: E731
    if mode == "hybrid_rerank" and reranker is None:
        reranker = CrossEncoderReranker()
    return Retriever(
        mode,
        dense_search=dense_search,
        bm25=bm25,
        reranker=reranker,
        text_of=text_of,
        **kwargs,
    )
