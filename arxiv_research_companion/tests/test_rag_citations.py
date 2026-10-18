"""Citation enforcement, refusal and 'sources survive LLM failure' in rag_engine."""
import unittest
from unittest.mock import patch

import numpy as np

from arxiv_research_companion import rag_engine
from arxiv_research_companion.embedders import HashEmbedder
from arxiv_research_companion.llm_http import LlmHttpError
from arxiv_research_companion.retrieval import DenseIndex, build_retriever, doc_text

PAPERS = [
    {"arxiv_id": "2610.00001", "title": "Sparse attention", "authors": ["A"], "summary": "Sparse attention reduces the memory cost of long context transformers."},
    {"arxiv_id": "2610.00002", "title": "Diffusion", "authors": ["B"], "summary": "Diffusion models generate images by iterative denoising."},
    {"arxiv_id": "2610.00003", "title": "Robot RL", "authors": ["C"], "summary": "Reinforcement learning lets robots grasp objects."},
]


def make_store():
    emb = HashEmbedder()
    idx = DenseIndex([p["arxiv_id"] for p in PAPERS], emb.encode([doc_text(p) for p in PAPERS]))
    retriever = build_retriever("dense", embedder=emb, papers=PAPERS, dense_index=idx)
    return {"model": emb, "collection": None, "retriever": retriever,
            "papers_by_id": {p["arxiv_id"]: p for p in PAPERS}}


def cfg(**citations):
    base = {"enforce": True, "policy": "drop", "max_repair_attempts": 1}
    base.update(citations)
    return {"retrieval_k": 3, "citations": base,
            "llm": {"provider": "groq", "groq_api_key": "k", "model_name": "m", "max_retries": 0}}


class Llm:
    """Scripted replacement for _call_groq_messages."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, **kw):
        self.calls.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class TestEnforcement(unittest.TestCase):
    def run_engine(self, llm, **citations):
        with patch.object(rag_engine, "_call_groq_messages", llm):
            return rag_engine.generate_answer("sparse attention memory", make_store(), cfg(**citations))

    def test_compliant_first_try_no_repair(self):
        llm = Llm("Sparse attention lowers memory cost for long inputs [1].")
        res = self.run_engine(llm)
        self.assertEqual(len(llm.calls), 1)
        self.assertTrue(res["citations"]["first_pass_ok"])
        self.assertEqual(res["citations"]["cited_ids"], ["2610.00001"])
        self.assertFalse(res["degraded"])
        self.assertEqual(len(res["sources"]), 3)

    def test_prompt_carries_numbering_and_contract(self):
        llm = Llm("Fine and properly cited statement here [1].")
        self.run_engine(llm)
        prompt = llm.calls[0][0]["content"]
        self.assertIn("[1] arXiv:", prompt)
        self.assertIn("[3] arXiv:", prompt)
        self.assertIn("NOT_ENOUGH_INFORMATION", prompt)
        self.assertIn("1 to 3", prompt)

    def test_repair_round_fixes_violation(self):
        llm = Llm("Uncited claim about sparse attention memory use.",
                  "Sparse attention lowers memory cost for long inputs [1].")
        res = self.run_engine(llm)
        self.assertEqual(len(llm.calls), 2)
        follow_up = llm.calls[1]
        self.assertEqual([m["role"] for m in follow_up], ["user", "assistant", "user"])
        self.assertIn("uncited", follow_up[2]["content"])
        self.assertFalse(res["citations"]["first_pass_ok"])
        self.assertTrue(res["citations"]["ok_after_repair"])
        self.assertEqual(res["citations"]["repair_attempts"], 1)
        self.assertEqual(res["answer"], "Sparse attention lowers memory cost for long inputs [1].")
        self.assertIn("raw_answer", res)

    def test_unrepaired_uncited_sentences_are_dropped(self):
        bad = "Sparse attention lowers memory cost [1]. This extra claim has no citation whatsoever."
        res = self.run_engine(Llm(bad, bad))
        self.assertEqual(res["answer"], "Sparse attention lowers memory cost [1].")
        self.assertEqual(len(res["citations"]["dropped_sentences"]), 1)

    def test_out_of_range_citation_is_removed(self):
        llm = Llm("Claim one is cited properly here [1][7].", "Claim one is cited properly here [1][7].")
        res = self.run_engine(llm)
        self.assertEqual(res["answer"], "Claim one is cited properly here [1].")

    def test_nothing_citable_degrades_but_keeps_sources(self):
        res = self.run_engine(Llm("No citations anywhere in this answer.", "Still none at all in this answer."))
        self.assertTrue(res["degraded"])
        self.assertEqual(res["answer"], rag_engine.UNCITED_MESSAGE)
        self.assertEqual(len(res["sources"]), 3)

    def test_refusal_sentinel(self):
        llm = Llm("NOT_ENOUGH_INFORMATION")
        res = self.run_engine(llm)
        self.assertTrue(res["refused"])
        self.assertEqual(len(llm.calls), 1)  # a refusal is never 'repaired'
        self.assertEqual(res["answer"], rag_engine.REFUSAL_MESSAGE)
        self.assertEqual(len(res["sources"]), 3)

    def test_repair_call_failure_falls_back_to_enforcement(self):
        first = "Sparse attention lowers memory cost [1]. Extra claim without any citation here."
        res = self.run_engine(Llm(first, LlmHttpError("HTTP 503", 503, 4)))
        self.assertEqual(res["answer"], "Sparse attention lowers memory cost [1].")
        self.assertFalse(res["degraded"])

    def test_enforcement_off_returns_raw_text(self):
        res = self.run_engine(Llm("whatever the model said"), enforce=False)
        self.assertEqual(res["answer"], "whatever the model said")
        self.assertIsNone(res["citations"])


class TestSourcesSurviveFailures(unittest.TestCase):
    def run_failing(self, error, **kw):
        llm = Llm(error)
        with patch.object(rag_engine, "_call_groq_messages", llm):
            return rag_engine.generate_answer("sparse attention memory", make_store(), cfg(**kw))

    def test_http_429_after_retries(self):
        res = self.run_failing(LlmHttpError("HTTP 429", 429, 4))
        self.assertTrue(res["degraded"])
        self.assertEqual(len(res["sources"]), 3)
        self.assertIn("rate-limiting", res["answer"])
        self.assertIn("Sparse attention", res["answer"])  # abstracts shown in the answer text

    def test_http_503_after_retries(self):
        res = self.run_failing(LlmHttpError("HTTP 503", 503, 4))
        self.assertEqual(len(res["sources"]), 3)
        self.assertIn("HTTP 503", res["answer"])

    def test_missing_key(self):
        res = self.run_failing(ValueError("GROQ_API_KEY is not set"))
        self.assertEqual(len(res["sources"]), 3)
        self.assertIn("No Groq API key", res["answer"])

    def test_unexpected_exception(self):
        res = self.run_failing(RuntimeError("boom"))
        self.assertEqual(len(res["sources"]), 3)
        self.assertTrue(res["llm_error"].startswith("RuntimeError"))

    def test_retrieve_sources_helper_needs_no_llm(self):
        sources = rag_engine.retrieve_sources("diffusion images", make_store(), 2)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["id"], "2610.00002")
        self.assertIn("denoising", sources[0]["document"])


if __name__ == "__main__":
    unittest.main()
