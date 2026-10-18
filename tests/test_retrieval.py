import unittest

import numpy as np

from arxiv_research_companion.embedders import HashEmbedder
from arxiv_research_companion.retrieval import (
    Bm25Index,
    DenseIndex,
    Retriever,
    build_retriever,
    doc_text,
    fts_query,
    make_numpy_dense_search,
    rrf_fuse,
)

PAPERS = [
    {"arxiv_id": "p1", "title": "Diffusion models", "summary": "We study diffusion models for image generation and denoising."},
    {"arxiv_id": "p2", "title": "Robot RL", "summary": "Reinforcement learning enables robots to grasp objects with policies."},
    {"arxiv_id": "p3", "title": "Long context", "summary": "Sparse attention reduces the memory cost of long context transformers."},
    {"arxiv_id": "p4", "title": "Graph nets", "summary": "Graph neural networks propagate messages along edges of a graph."},
]


class FakeReranker:
    """Scores by word overlap with the query (deterministic, torch-free)."""

    def score(self, query, texts):
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


class TestPieces(unittest.TestCase):
    def test_fts_query_quotes_tokens_and_drops_stopwords(self):
        self.assertEqual(fts_query("What is the NEAR operator?"), '"near" OR "operator"')
        self.assertEqual(fts_query("???"), "")
        self.assertEqual(fts_query("sparse-attention sparse"), '"sparse" OR "attention"')

    def test_bm25_ranks_keyword_match_first_and_survives_reserved_words(self):
        idx = Bm25Index([(p["arxiv_id"], doc_text(p)) for p in PAPERS])
        hits = idx.search("diffusion models for image generation", 3)
        self.assertEqual(hits[0][0], "p1")
        self.assertTrue(all(s > 0 for _, s in hits))
        idx.search('NEAR "unbalanced AND (', 3)  # must not raise
        self.assertEqual(idx.search("zzzzqqq", 3), [])

    def test_bm25_stemming(self):
        idx = Bm25Index([(p["arxiv_id"], doc_text(p)) for p in PAPERS])
        self.assertEqual(idx.search("robot grasping policy", 1)[0][0], "p2")

    def test_rrf_by_hand(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
        self.assertEqual([d for d, _ in fused], ["a", "b", "c", "d"])  # ties -> doc_id order
        self.assertAlmostEqual(dict(fused)["a"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(dict(fused)["c"], 1 / 63)

    def test_rrf_ignores_duplicates_within_a_list(self):
        fused = dict(rrf_fuse([["a", "a", "b"]], k=60))
        self.assertAlmostEqual(fused["a"], 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 63)

    def test_dense_index_exact_and_deterministic_ties(self):
        m = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        idx = DenseIndex(["b", "a", "c"], m)
        res = idx.search_vector(np.array([1.0, 0.0]), 3)
        self.assertEqual([d for d, _ in res], ["a", "b", "c"])
        self.assertAlmostEqual(res[0][1], 1.0, places=6)
        self.assertAlmostEqual(res[2][1], 0.0, places=6)

    def test_doc_text_modes(self):
        self.assertEqual(doc_text(PAPERS[0], "abstract"), PAPERS[0]["summary"])
        self.assertTrue(doc_text(PAPERS[0], "title_abstract").startswith("Diffusion models."))
        with self.assertRaises(ValueError):
            doc_text(PAPERS[0], "nope")


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.emb = HashEmbedder()
        texts = [doc_text(p) for p in PAPERS]
        self.index = DenseIndex([p["arxiv_id"] for p in PAPERS], self.emb.encode(texts))

    def _mk(self, mode, **kw):
        return build_retriever(
            mode, embedder=self.emb, papers=PAPERS, dense_index=self.index,
            reranker=FakeReranker() if mode == "hybrid_rerank" else None, **kw
        )

    def test_all_modes_find_the_obvious_paper(self):
        for mode in ("dense", "bm25", "hybrid", "hybrid_rerank"):
            hits = self._mk(mode).retrieve("sparse attention for long context", 3)
            self.assertEqual(hits[0].doc_id, "p3", mode)
            self.assertEqual([h.rank for h in hits], [1, 2, 3][: len(hits)])

    def test_scores_are_non_increasing(self):
        for mode in ("dense", "bm25", "hybrid", "hybrid_rerank"):
            hits = self._mk(mode).retrieve("graph neural network edges", 4)
            scores = [h.score for h in hits]
            self.assertEqual(scores, sorted(scores, reverse=True), mode)

    def test_rerank_tail_keeps_rrf_order_below_reranked_head(self):
        r = self._mk("hybrid_rerank", rerank_depth=2, candidate_depth=4)
        hits = r.retrieve("diffusion image", 4)
        self.assertEqual(len(hits), 4)
        head, tail = hits[:2], hits[2:]
        self.assertGreater(min(h.score for h in head), max(h.score for h in tail))
        self.assertEqual(len({h.doc_id for h in hits}), 4)

    def test_empty_query_and_bad_mode(self):
        self.assertEqual(self._mk("dense").retrieve("   ", 3), [])
        with self.assertRaises(ValueError):
            Retriever("nope")
        with self.assertRaises(ValueError):
            Retriever("hybrid", dense_search=lambda q, k: [])  # bm25 missing

    def test_dense_top1_score(self):
        r = self._mk("dense")
        s = r.dense_top1_score("diffusion models image generation denoising")
        self.assertTrue(0.0 < s <= 1.0 + 1e-6)
        self.assertIsNone(self._mk("bm25").dense_top1_score("x"))


if __name__ == "__main__":
    unittest.main()
