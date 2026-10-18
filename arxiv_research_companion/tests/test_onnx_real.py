"""Slow/network: exercises the REAL chromadb ONNX model (downloads ~80 MB on first use).

Run with:  uv run pytest -m slow
"""
import numpy as np
import pytest

pytest.importorskip("chromadb")

from arxiv_research_companion.embedders import OnnxEmbedder  # noqa: E402


@pytest.mark.slow
@pytest.mark.network
def test_real_onnx_embedder_shape_and_semantics():
    emb = OnnxEmbedder()
    v = emb.encode([
        "Diffusion models generate images by iterative denoising.",
        "Image synthesis with denoising diffusion.",
        "Reinforcement learning for robot grasping.",
    ])
    assert v.shape == (3, 384)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2])
