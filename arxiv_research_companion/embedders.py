"""
Embedding back-ends with one interface: ``encode(list[str]) -> np.ndarray[float32]``.

``OnnxEmbedder``             what the Streamlit app deploys (chromadb's bundled ONNX
                             all-MiniLM-L6-v2, no PyTorch)
``SentenceTransformerEmbedder``  reference implementation (needs the ``eval`` group)
``HashEmbedder``             deterministic, dependency-free, **for tests/smoke runs only**
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Sequence

import numpy as np

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class OnnxEmbedder:
    """Lightweight stand-in for SentenceTransformer (no PyTorch needed)."""

    name = "onnx-minilm-l6-v2"

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2()

    def encode(self, texts, show_progress_bar=False, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.array(
            [np.asarray(v, dtype="float32") for v in self._fn(list(texts))]
        )

    def get_sentence_embedding_dimension(self) -> int:
        return 384


class SentenceTransformerEmbedder:
    name = "sentence-transformers-minilm-l6-v2"

    def __init__(self, model_name: str = ST_MODEL, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts, show_progress_bar=False, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
            ),
            dtype="float32",
        )

    def get_sentence_embedding_dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Bag-of-hashed-tokens embedding.  Not a semantic model: tests/smoke runs only."""

    name = "hash-smoke"
    is_smoke = True

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: Sequence[str], show_progress_bar=False, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out: List[np.ndarray] = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in _TOKEN.findall(text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "big")
                v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim
