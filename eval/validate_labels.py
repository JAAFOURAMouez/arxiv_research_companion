"""
Sanity-check the hand-labelled question set BEFORE running anything.

    python -m eval.validate_labels --questions eval/data/questions.csv --db arxiv_papers.db
    python -m eval.validate_labels ... --check-arxiv         # also resolve IDs on arxiv.org
    python -m eval.validate_labels ... --ooc eval/data/ooc_questions.csv   # audit OOC questions

Reports how many labelled papers exist in the production corpus.  If that coverage is low the
"prod" retrieval numbers will mostly measure the labelling, not the retriever (see eval/README.md).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from eval.build_corpus import read_db
from eval.common import is_valid_arxiv_id, load_questions, write_json, RESULTS_DIR


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="eval/data/questions.csv")
    ap.add_argument("--db", default="arxiv_papers.db")
    ap.add_argument("--expected", type=int, default=40)
    ap.add_argument("--check-arxiv", action="store_true")
    ap.add_argument("--ooc", default=None)
    ap.add_argument("--out", default=str(RESULTS_DIR / "label_validation.json"))
    args = ap.parse_args(argv)

    errors, warnings = [], []
    questions = load_questions(args.questions)
    if len(questions) != args.expected:
        warnings.append(f"{len(questions)} questions found, expected {args.expected}")
    qids = Counter(q["qid"] for q in questions)
    errors += [f"duplicate qid {k}" for k, v in qids.items() if v > 1]
    texts = Counter(q["question"].lower() for q in questions)
    warnings += [f"duplicate question text: {k[:60]!r}" for k, v in texts.items() if v > 1]

    prod_ids = {p["arxiv_id"] for p in read_db(args.db)}
    per_q = []
    for q in questions:
        ids = q["relevant_arxiv_ids"]
        if not ids:
            errors.append(f"{q['qid']}: no relevant_arxiv_ids")
        bad = [i for i in ids if not is_valid_arxiv_id(i)]
        errors += [f"{q['qid']}: malformed arXiv ID {i!r}" for i in bad]
        per_q.append({"qid": q["qid"], "n_relevant": len(ids), "n_in_prod": sum(i in prod_ids for i in ids)})

    all_ids = sorted({i for q in questions for i in q["relevant_arxiv_ids"]})
    in_prod = [i for i in all_ids if i in prod_ids]
    summary = {
        "n_questions": len(questions), "n_labeled_ids": len(all_ids),
        "n_labeled_ids_in_prod": len(in_prod),
        "n_questions_with_any_relevant_in_prod": sum(1 for r in per_q if r["n_in_prod"] > 0),
        "n_questions_fully_covered_in_prod": sum(1 for r in per_q if r["n_in_prod"] == r["n_relevant"] > 0),
        "n_prod_papers": len(prod_ids),
    }
    if summary["n_questions_with_any_relevant_in_prod"] < min(30, len(questions)):
        warnings.append(
            f"only {summary['n_questions_with_any_relevant_in_prod']} questions have a relevant paper in the "
            f"production corpus; use `build_corpus --fetch-missing` and read the 'augmented' numbers, "
            f"and treat the 'prod' numbers as coverage-limited")

    if args.check_arxiv:
        from eval.arxiv_api import resolve_ids
        status = resolve_ids(all_ids, cache_path=RESULTS_DIR / "cache" / "arxiv_resolve.json")
        summary["arxiv_status"] = dict(Counter(status.values()))
        errors += [f"labelled ID does not resolve on arXiv: {i}" for i, s in status.items() if s == "not_found"]
        warnings += [f"could not check {i} (arXiv unreachable)" for i, s in status.items() if s == "unknown"]

    ooc_report = []
    if args.ooc:
        from arxiv_research_companion.retrieval import Bm25Index
        papers = read_db(args.db)
        titles = {p["arxiv_id"]: p["title"] for p in papers}
        idx = Bm25Index([(p["arxiv_id"], p["title"] + " " + p["summary"]) for p in papers])
        oocs = load_questions(args.ooc)
        for q in oocs:
            hits = idx.search(q["question"], 5)
            ooc_report.append({"qid": q["qid"], "question": q["question"],
                               "audited": str(q.get("audited", "")).lower() in ("1", "true", "yes"),
                               "top_bm25_titles": [titles[d] for d, _ in hits]})
        n_aud = sum(r["audited"] for r in ooc_report)
        summary["ooc"] = {"n": len(ooc_report), "n_audited": n_aud}
        if n_aud < len(ooc_report):
            warnings.append(f"{len(ooc_report) - n_aud} OOC questions not marked audited=true; "
                            "refusal_eval only counts audited ones")
        print("\nOOC audit: read the titles below; if ANY paper could answer the question, delete it\n"
              "from the OOC file (it is not out-of-corpus), then set audited=true.\n")
        for r in ooc_report:
            print(f"[{r['qid']}] {r['question']}")
            for t in r["top_bm25_titles"]:
                print(f"     - {t[:110]}")

    write_json(args.out, {"summary": summary, "per_question": per_q, "errors": errors,
                          "warnings": warnings, "ooc_audit": ooc_report})
    print("\nSUMMARY", summary)
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
