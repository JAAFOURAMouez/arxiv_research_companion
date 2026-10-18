import unittest

from arxiv_research_companion import citations as c

IDS = ["2610.00001", "2610.00002", "2610.00003"]


class TestParsing(unittest.TestCase):
    def test_extract_numbers_forms(self):
        self.assertEqual(c.extract_numbers("A [1]. B [2][3]. C [1, 3]. D [2-3]."), [1, 2, 3])
        self.assertEqual(c.extract_numbers("no markers, list [a] and (1)"), [])

    def test_sentence_split_keeps_markers_and_protects_abbreviations(self):
        text = "It works e.g. on 3.5B models [2][3]. Results improve by 4.2% [1]. [2] Next sentence starts here."
        out = c.split_sentences(text)
        self.assertEqual(out[0], "It works e.g. on 3.5B models [2][3].")
        self.assertEqual(out[1], "Results improve by 4.2% [1]. [2]")  # trailing marker stays with its sentence
        self.assertEqual(out[2], "Next sentence starts here.")

    def test_bullets_and_newlines(self):
        out = c.split_sentences("- First bullet has enough words here [1].\n- Second bullet also has enough words [2].")
        self.assertEqual(len(out), 2)
        self.assertFalse(out[0].startswith("-"))

    def test_strip_markers(self):
        self.assertEqual(c.strip_citation_markers("A thing [1][2]. Another [3]."), "A thing. Another.")

    def test_refusal(self):
        self.assertTrue(c.is_refusal("NOT_ENOUGH_INFORMATION"))
        self.assertTrue(c.is_refusal('  "NOT_ENOUGH_INFORMATION".'))
        self.assertFalse(c.is_refusal("Diffusion models are good [1]."))


class TestValidate(unittest.TestCase):
    def test_clean_answer(self):
        r = c.validate_answer("Sparse attention cuts memory use [1]. It also speeds training up [2][3].", IDS)
        self.assertTrue(r.ok)
        self.assertEqual(r.cited_ids, IDS)

    def test_uncited_out_of_range_and_inline_ids(self):
        ans = ("This sentence has no citation at all in it. "
               "Cited out of range here [9]. "
               "See arXiv 2610.99999 for the details of the method [1].")
        r = c.validate_answer(ans, IDS)
        types = {v["type"] for v in r.violations}
        self.assertEqual(types, {"uncited", "out_of_range", "unknown_arxiv_id"})

    def test_retrieved_id_typed_inline_is_allowed(self):
        r = c.validate_answer("See 2610.00002v2 for more details on the method [2].", IDS)
        self.assertTrue(r.ok)

    def test_short_fragments_do_not_need_citation(self):
        r = c.validate_answer("Summary:\nThe method reduces memory use a lot [1].", IDS)
        self.assertTrue(r.ok)

    def test_refusal_is_not_a_violation(self):
        r = c.validate_answer("NOT_ENOUGH_INFORMATION", IDS)
        self.assertTrue(r.ok and r.refused)

    def test_empty_answer_is_violation(self):
        self.assertFalse(c.validate_answer("   ", IDS).ok)


class TestEnforce(unittest.TestCase):
    def test_drop_policy(self):
        ans = "Good claim with a citation [1]. Bad claim with none at all here. Another good one [2][9]."
        text, info = c.enforce(ans, IDS, "drop")
        self.assertEqual(text, "Good claim with a citation [1]. Another good one [2].")
        self.assertEqual(info["dropped_sentences"], ["Bad claim with none at all here."])
        # what is left satisfies the contract
        self.assertTrue(c.validate_answer(text, IDS).ok)

    def test_flag_policy(self):
        text, info = c.enforce("Bad claim with none at all here.", IDS, "flag")
        self.assertTrue(text.endswith("[uncited]"))
        self.assertEqual(len(info["flagged_sentences"]), 1)

    def test_everything_dropped_gives_empty(self):
        text, info = c.enforce("Nothing is cited in this whole answer.", IDS, "drop")
        self.assertEqual(text, "")
        self.assertTrue(info["empty_after_enforcement"])

    def test_structure_preserved_and_markers_normalised(self):
        ans = "- Point one is a real claim here [1, 2].\n- Point two is another real claim [3]."
        text, _ = c.enforce(ans, IDS)
        self.assertEqual(text, "- Point one is a real claim here [1][2].\n- Point two is another real claim [3].")

    def test_bad_policy(self):
        with self.assertRaises(ValueError):
            c.enforce("x", IDS, "nope")


if __name__ == "__main__":
    unittest.main()
