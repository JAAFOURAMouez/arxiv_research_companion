"""
Freeze the corpus used for evaluation.

The production database grows every night (new distractors) and users may add papers, so
retrieval numbers are only comparable if they are computed on a frozen snapshot.  This writes
`corpus_snapshot.jsonl` plus a `.meta.json` with SHA-256 hashes; every result file records the
snapshot hash.

Labelled papers that are not in the production database cannot be retrieved by the app as it
stands.  With --fetch-missing they are downloaded from the arXiv API and added with
origin="labeled_extra" so the "augmented" corpus mode (all labelled positives present among the
production papers as distractors) can be evaluated next to the "prod" mode.

    python -m eval.build_corpus --db arxiv_papers.db --questions eval/data/questions.csv \
        --out eval/data/corpus_snapshot.jsonl --fetch-missing
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval.common import load_questions, sha256_file, write_json, write_jsonl


def read_db(db_path: str):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT arxiv_id, title, summary, categories, published FROM papers ORDER BY arxiv_id"
        ).fetchall()
    finally:
        conn.close()
    return [{"arxiv_id": a, "title": t or "", "summary": s or "",
             "categories": json.loads(c) if c else [], "published": p, "origin": "prod"}
            for a, t, s, c, p in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="arxiv_papers.db")
    ap.add_argument("--questions", default="eval/data/questions.csv")
    ap.add_argument("--out", default="eval/data/corpus_snapshot.jsonl")
    ap.add_argument("--fetch-missing", action="store_true",
                    help="download labelled papers absent from the DB from the arXiv API")
    args = ap.parse_args(argv)

    prod = read_db(args.db)
    have = {p["arxiv_id"] for p in prod}
    questions = load_questions(args.questions)
    labeled = [i for q in questions for i in q["relevant_arxiv_ids"]]
    missing = sorted({i for i in labeled if i not in have})
    extra, unresolved = [], []
    if missing and args.fetch_missing:
        from eval.arxiv_api import fetch_by_ids
        print(f"fetching {len(missing)} labelled papers missing from the DB ...", flush=True)
        fetched = fetch_by_ids(missing)
        for i, entry in fetched.items():
            if entry is None:
                unresolved.append(i)
            else:
                extra.append({**entry, "origin": "labeled_extra"})
    elif missing:
        unresolved = missing
        print(f"NOTE: {len(missing)} labelled IDs are not in the DB; rerun with --fetch-missing "
              f"to build the 'augmented' corpus.", file=sys.stderr)

    rows = prod + sorted(extra, key=lambda r: r["arxiv_id"])
    write_jsonl(args.out, rows)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_sha256": sha256_file(args.out), "db_sha256": sha256_file(args.db),
        "n_prod": len(prod), "n_labeled_extra": len(extra),
        "labeled_ids_total": len(set(labeled)), "labeled_ids_in_prod": len(set(labeled) & have),
        "labeled_ids_unresolved": unresolved,
    }
    write_json(Path(args.out).with_suffix(".meta.json"), meta)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
