"""End-to-end run of the generation-side scripts with a scripted LLM and the hash embedder.
Judges are not exercised here (they need torch / network); their math is in test_eval_units."""
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arxiv_research_companion import rag_engine
from eval import (calibrate_judge, check_citations, generate_answers, make_calibration_set, refusal_eval)
from eval.common import read_json, read_jsonl, write_jsonl

DOCS = [
    ("2610.00001", "Sparse attention", "Sparse attention reduces the memory cost of long context transformers. It keeps accuracy on long documents."),
    ("2610.00002", "Diffusion", "Diffusion models generate images by iterative denoising. They need many sampling steps."),
    ("2610.00003", "Robot RL", "Reinforcement learning lets robots grasp objects. Policies are trained in simulation."),
    ("2610.00004", "Graph nets", "Graph neural networks propagate messages along edges. They are used for molecules."),
    ("2610.00005", "Speech", "Speech models transcribe audio into text. They are trained on hours of recordings."),
]


class FakeLlm:
    def __call__(self, messages, **kw):
        prompt = messages[0]["content"]
        question = prompt.split("Question:")[1].split("Answer:")[0].strip().lower()
        if "capital" in question:
            return "NOT_ENOUGH_INFORMATION"
        return "Sparse attention reduces memory cost [1]. It also keeps accuracy on long documents [1][2]."


class TestGenerationChain(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.snap = self.tmp / "snap.jsonl"
        write_jsonl(self.snap, [{"arxiv_id": a, "title": t, "summary": s, "origin": "prod"} for a, t, s in DOCS])
        with open(self.tmp / "q.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["qid", "question", "relevant_arxiv_ids"])
            w.writerow(["q01", "How does sparse attention cut memory?", "2610.00001"])
            w.writerow(["q02", "How do diffusion models generate images?", "2610.00002"])
            w.writerow(["q03", "How do robots learn to grasp?", "2610.00003"])
        with open(self.tmp / "ooc.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["qid", "question", "relevant_arxiv_ids", "kind", "audited"])
            w.writerow(["o01", "What is the capital of Australia?", "", "off_topic", "true"])
            w.writerow(["o02", "Unaudited capital question?", "", "off_topic", ""])
        self.answers = self.tmp / "answers.jsonl"
        env = {"GROQ_API_KEY": "x", "LLM_PROVIDER": "groq"}
        with patch.dict(os.environ, env), patch.object(rag_engine, "_call_groq_messages", FakeLlm()):
            code = generate_answers.main([
                "--questions", str(self.tmp / "q.csv"), "--ooc", str(self.tmp / "ooc.csv"),
                "--snapshot", str(self.snap), "--embedder", "hash", "--sleep", "0",
                "--retrieval-mode", "hybrid", "--out", str(self.answers)])
        self.assertEqual(code, 0)

    def test_answers_file_contents_and_smoke_flag(self):
        recs = read_jsonl(self.answers)
        self.assertEqual([r["qid"] for r in recs], ["q01", "q02", "q03", "o01", "o02"])
        self.assertTrue(recs[0]["citations"]["first_pass_ok"])
        self.assertTrue(recs[3]["refused"])
        self.assertEqual(len(recs[0]["retrieved_ids"]), 3)
        self.assertTrue(read_json(self.answers.with_suffix(".meta.json"))["smoke"])  # hash embedder => smoke

    def test_resume_skips_done(self):
        before = self.answers.read_text()
        with patch.dict(os.environ, {"GROQ_API_KEY": "x", "LLM_PROVIDER": "groq"}), \
                patch.object(rag_engine, "_call_groq_messages", FakeLlm()):
            generate_answers.main(["--questions", str(self.tmp / "q.csv"), "--snapshot", str(self.snap),
                                   "--embedder", "hash", "--sleep", "0", "--resume", "--out", str(self.answers)])
        self.assertEqual(self.answers.read_text(), before)

    def test_check_citations_without_judge(self):
        out = self.tmp / "cc.json"
        check_citations.main(["--answers", str(self.answers), "--snapshot", str(self.snap),
                              "--judge", "none", "--calibration", str(self.tmp / "none.json"), "--out", str(out)])
        res = read_json(out)
        self.assertEqual(res["format"]["n_answered"], 3)
        self.assertEqual(res["n_refused_excluded"], 0)  # the refusal is an OOC record, not in-scope
        self.assertEqual(res["final"]["with_valid_citation"]["p"], 1.0)
        self.assertEqual(res["final"]["cited_id_in_retrieved"]["p"], 1.0)
        self.assertFalse(res["meta"]["calibrated"])

    def test_refusal_eval_counts_only_audited(self):
        out = self.tmp / "rf.json"
        self.assertEqual(refusal_eval.main(["--answers", str(self.answers), "--out", str(out)]), 0)
        res = read_json(out)
        self.assertEqual(res["ooc_refusal"]["n"], 1)
        self.assertEqual(res["ooc_refusal"]["k"], 1)
        self.assertEqual(res["in_scope_false_refusal"]["k"], 0)
        self.assertEqual(res["n_ooc_unaudited_excluded"], 1)

    def test_calibration_set_hides_kind_and_has_blank_labels(self):
        out, key = self.tmp / "cal.csv", self.tmp / "key.json"
        make_calibration_set.main(["--answers", str(self.answers), "--snapshot", str(self.snap),
                                   "--n", "6", "--n-swapped", "2", "--out", str(out), "--key", str(key)])
        rows = list(csv.DictReader(open(out, newline="")))
        self.assertEqual(len(rows), 6)
        self.assertNotIn("kind", rows[0])
        self.assertTrue(all(r["label"] == "" for r in rows))
        kinds = [v["kind"] for v in read_json(key).values()]
        self.assertEqual(sorted(set(kinds)), ["real", "swapped"])
        for r in rows:  # no judge score leaks into the labelling sheet
            self.assertNotIn("nli", " ".join(r.keys()).lower())

    def test_calibrate_rejects_unlabelled_sheet(self):
        out, key = self.tmp / "cal.csv", self.tmp / "key.json"
        make_calibration_set.main(["--answers", str(self.answers), "--snapshot", str(self.snap),
                                   "--n", "4", "--n-swapped", "1", "--out", str(out), "--key", str(key)])
        self.assertEqual(calibrate_judge.main(["--labels", str(out), "--snapshot", str(self.snap),
                                               "--out", str(self.tmp / "c.json"), "--key", str(key)]), 1)


if __name__ == "__main__":
    unittest.main()
