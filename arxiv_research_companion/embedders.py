"""
Embedding back-ends with one interface: ``encode(list[str]) -> np.ndarray[float32]``.

``OnnxEmbedder``             what the Streamlit app deploys (chromadb's bundled ONNX
                             all-MiniLM-L6-v2, no PyTorch)
``SentenceTransformerEmbedder``  reference implementation (needs the ``eval`` group)
``HashEmbedder``             deterministic, dependency-free, **for tests/smoke runs only**
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class OnnxEmbedder:
    """Lightweight stand-in for SentenceTransformer (no PyTorch needed).

    Memory: with chromadb's default session (ONNX Runtime CPU memory arena, batches of 32)
    embedding the corpus peaks around 760 MB, over the 512 MB of Render's free instance.
    Without the arena and with small batches it peaks around 350 MB, same vectors, same speed.
    Tune with ONNX_BATCH_SIZE / ONNX_THREADS.
    """

    name = "onnx-minilm-l6-v2"

    def __init__(self, batch_size: Optional[int] = None, threads: Optional[int] = None):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self.batch_size = batch_size or int(os.getenv("ONNX_BATCH_SIZE", "8"))
        self._fn = ONNXMiniLM_L6_V2()
        model_dir = os.getenv("ONNX_MODEL_DIR")  # keep the model inside the deployed project dir
        if model_dir:
            self._fn.DOWNLOAD_PATH = Path(model_dir).resolve()
        self._lean_session(threads if threads is not None else int(os.getenv("ONNX_THREADS", "1")))

    def _lean_session(self, threads: int) -> None:
        """Replace chromadb's lazily built session with one that has no CPU memory arena."""
        fn = self._fn
        needed = ("_download_model_if_not_exists", "DOWNLOAD_PATH", "EXTRACTED_FOLDER_NAME", "_forward")
        if not all(hasattr(fn, attr) for attr in needed):  # test doubles, other chromadb versions
            return
        try:
            import onnxruntime as ort

            fn._download_model_if_not_exists()
            options = ort.SessionOptions()
            options.log_severity_level = 3
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_cpu_mem_arena = False
            if threads > 0:
                options.intra_op_num_threads = threads
            fn.__dict__["model"] = ort.InferenceSession(  # `model` is a cached_property
                os.path.join(fn.DOWNLOAD_PATH, fn.EXTRACTED_FOLDER_NAME, "model.onnx"),
                options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # fall back to chromadb's own session
            logger.warning("Could not build a low-memory ONNX session (%s); using the default", exc)

    def encode(self, texts, show_progress_bar=False, **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if hasattr(self._fn, "_forward"):
            vectors = self._fn._forward(texts, batch_size=self.batch_size)
        else:
            vectors = self._fn(texts)
        return np.array([np.asarray(v, dtype="float32") for v in vectors])

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
