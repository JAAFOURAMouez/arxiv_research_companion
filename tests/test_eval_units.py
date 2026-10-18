import math
import os
import tempfile
import unittest

from eval import metrics as M
from eval.arxiv_api import parse_feed
from eval.calibrate_judge import best_threshold, binary_report, loo_eval
from eval.common import is_valid_arxiv_id, load_questions, normalize_arxiv_id
from eval.judges import premise_windows


class TestMetricsByHand(unittest.TestCase):
    R = ["a", "b", "c", "d", "e"]

    def test_single_relevant_at_rank_2(self):
        m = M.per_query(self.R, {"b"})
        self.assertEqual((m["recall@1"], m["recall@3"], m["mrr@10"]), (0.0, 1.0, 0.5))
        self.assertAlmostEqual(m["ndcg@10"], 1 / math.log2(3))

    def test_two_relevant(self):
        m = M.per_query(self.R, {"a", "c"})
        self.assertEqual(m["recall@1"], 0.5)  # recall is bounded by 1/|relevant| at k=1
        self.assertEqual(m["hit@1"], 1.0)
        self.assertAlmostEqual(m["ndcg@10"], (1 + 0.5) / (1 + 1 / math.log2(3)))

    def test_miss_and_perfect(self):
        self.assertEqual(M.per_query(self.R, {"z"})["ndcg@10"], 0.0)
        self.assertAlmostEqual(M.per_query(self.R, {"a"})["ndcg@10"], 1.0)
        # relevant doc outside top-10 gives 0 for mrr@10 / ndcg@10 / recall@10
        deep = [str(i) for i in range(20)]
        m = M.per_query(deep, {"15"})
        self.assertEqual((m["mrr@10"], m["ndcg@10"], m["recall@10"]), (0.0, 0.0, 0.0))

    def test_reference_engine_aggregation(self):
        rank = {"q1": ["a", "b"], "q2": ["x", "a"]}
        ev = M.evaluate_run(rank, {"q1": {"a"}, "q2": {"a"}}, engine="reference")
        self.assertAlmostEqual(ev["means"]["mrr@10"], 0.75)
        self.assertAlmostEqual(ev["means"]["recall@1"], 0.5)

    def test_ranx_agrees_when_installed(self):
        try:
            import ranx  # noqa: F401
        except ImportError:
            self.skipTest("ranx not installed")
        rank = {"q1": ["a", "b", "c"], "q2": ["x", "a", "c"], "q3": ["m", "n", "o"]}
        qrels = {"q1": {"a"}, "q2": {"a", "c"}, "q3": {"zz"}}
        ev = M.evaluate_run(rank, qrels, engine="ranx")
        self.assertLess(ev["ranx_vs_reference_max_abs_diff"], 1e-6)


class TestStats(unittest.TestCase):
    def test_wilson_known_value(self):
        p, lo, hi = M.wilson(5, 10)
        self.assertEqual(p, 0.5)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_auc_and_kappa(self):
        self.assertEqual(M.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(M.auc([0.5, 0.5], [0, 1]), 0.5)
        self.assertEqual(M.auc([0.9, 0.1], [0, 1]), 0.0)
        self.assertEqual(M.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertAlmostEqual(M.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)

    def test_bootstrap_ci_brackets_mean_and_is_seeded(self):
        v = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        a = M.bootstrap_ci(v, n=2000, seed=3)
        self.assertEqual(a, M.bootstrap_ci(v, n=2000, seed=3))
        self.assertLess(a[1], a[0])
        self.assertGreater(a[2], a[0])

    def test_paired_bootstrap_detects_clear_difference(self):
        a = [1.0] * 30
        b = [0.0] * 30
        d = M.paired_bootstrap_diff(a, b, n=500)
        self.assertGreater(d["ci_lo"], 0)

    def test_random_baseline_matches_expectation(self):
        ids = [f"d{i}" for i in range(100)]
        rb = M.random_baseline(ids, {"q": {"d0"}}, n_perm=2000, seed=1)
        self.assertAlmostEqual(rb["q"]["recall@10"], 0.10, delta=0.02)


class TestSelection(unittest.TestCase):
    def test_ties_choose_cheapest(self):
        same = [0.5, 0.6, 0.4, 0.7] * 5
        s = M.select_config({"bm25": same, "dense": same, "hybrid": same, "hybrid_rerank": same}, n_boot=500)
        self.assertEqual(s["chosen"], "bm25")

    def test_clear_winner_is_chosen_over_cheaper(self):
        low = [0.2] * 40
        high = [0.9] * 40
        s = M.select_config({"bm25": low, "dense": low, "hybrid": low, "hybrid_rerank": high}, n_boot=500)
        self.assertEqual(s["best_by_mean"], "hybrid_rerank")
        self.assertEqual(s["chosen"], "hybrid_rerank")

    def test_cheaper_equivalent_wins_over_best(self):
        import random
        rng = random.Random(0)
        base = [rng.random() for _ in range(40)]
        # same mean up to +0.002, but noisy per-query differences: not distinguishable
        better = [x + (0.05 if i % 2 else -0.046) for i, x in enumerate(base)]
        s = M.select_config({"bm25": base, "dense": base, "hybrid": base, "hybrid_rerank": better}, n_boot=800)
        self.assertEqual(s["best_by_mean"], "hybrid_rerank")
        self.assertEqual(s["chosen"], "bm25")


class TestCalibrationMath(unittest.TestCase):
    def test_threshold_separates_clean_data(self):
        scores = [0.05, 0.1, 0.2, 0.8, 0.9, 0.95]
        y = [0, 0, 0, 1, 1, 1]
        bt = best_threshold(scores, y)
        self.assertTrue(0.2 < bt["threshold"] < 0.8)
        self.assertEqual(bt["j"], 1.0)
        rep = loo_eval(scores, y)
        self.assertEqual(rep["accuracy"], 1.0)

    def test_single_class_gives_no_threshold(self):
        self.assertIsNone(best_threshold([0.1, 0.9], [1, 1])["threshold"])

    def test_binary_report_counts(self):
        r = binary_report([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((r["tp"], r["fp"], r["fn"], r["tn"]), (1, 1, 1, 1))


class TestHelpers(unittest.TestCase):
    def test_normalize_ids(self):
        for raw in ["arXiv:2610.02207v2", "https://arxiv.org/abs/2610.02207v3", " 2610.02207 ", "https://arxiv.org/pdf/2610.02207.pdf"]:
            self.assertEqual(normalize_arxiv_id(raw), "2610.02207")
        self.assertTrue(is_valid_arxiv_id("2610.02207"))
        self.assertTrue(is_valid_arxiv_id("hep-th/9901001"))
        self.assertFalse(is_valid_arxiv_id("2610.22"))
        self.assertFalse(is_valid_arxiv_id("26100.02207"))

    def test_load_questions_csv(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("qid,question,relevant_arxiv_ids\nq01,What is X?,2610.00001v1; arXiv:2610.00002\nq02,,\n")
        try:
            qs = load_questions(fh.name)
        finally:
            os.unlink(fh.name)
        self.assertEqual(len(qs), 1)
        self.assertEqual(qs[0]["relevant_arxiv_ids"], ["2610.00001", "2610.00002"])

    def test_template_is_empty_so_nothing_runs_by_accident(self):
        root = os.path.dirname(os.path.dirname(__file__))
        self.assertEqual(load_questions(os.path.join(root, "eval/data/questions.csv")), [])

    def test_parse_feed_skips_error_entries(self):
        xml = """<feed xmlns="http://www.w3.org/2005/Atom">
          <entry><id>http://arxiv.org/abs/2610.00001v2</id><title> A  title </title><summary>S</summary>
            <published>2026-10-01T00:00:00Z</published><category term="cs.AI"/></entry>
          <entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_x</id><title>Error</title><summary>bad</summary></entry>
        </feed>"""
        got = parse_feed(xml)
        self.assertEqual(list(got), ["2610.00001"])
        self.assertEqual(got["2610.00001"]["title"], "A title")

    def test_premise_windows(self):
        abstract = "One is first. Two is second. Three is third. Four is fourth."
        w = premise_windows(abstract)
        self.assertIn(abstract, w)
        self.assertEqual(w[0], "One is first. Two is second. Three is third.")


if __name__ == "__main__":
    unittest.main()
