"""
Draw the 30 (sentence, cited paper) pairs you will label by hand to calibrate the judges.

    python -m eval.make_calibration_set --answers results/answers.jsonl \
        --snapshot eval/data/corpus_snapshot.jsonl

Writes eval/data/calibration_to_label.csv.  For each row read the ABSTRACT only (not any model
score; none is shown) and fill `label` with one of:
    supported    the abstract states or directly implies everything in the sentence
    partial      only part of the sentence is supported / it overstates the abstract
    unsupported  not supported or contradicted
The sample is mostly real (sentence, cited paper) pairs plus a few "swapped" pairs (the sentence
paired with a DIFFERENT retrieved paper) so that clear negatives exist.  Which rows are swapped is
kept in results/calibration_key.json, not in the CSV, so it cannot bias your labelling.
Real answers from a compliant model are mostly 'supported'; if you end up with fewer than ~8 rows
in either class the calibration script will warn that the threshold is unreliable.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys

from arxiv_research_companion import citations as C
from eval.common import RESULTS_DIR, load_snapshot, read_jsonl, write_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", default=str(RESULTS_DIR / "answers.jsonl"))
    ap.add_argument("--snapshot", default="eval/data/corpus_snapshot.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--n-swapped", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/data/calibration_to_label.csv")
    ap.add_argument("--key", default=str(RESULTS_DIR / "calibration_key.json"))
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    snap = {p["arxiv_id"]: p for p in load_snapshot(args.snapshot)}
    real_pool, swap_pool = [], []
    for r in read_jsonl(args.answers):
        if r["kind"] != "in_scope" or r["refused"] or r["degraded"]:
            continue
        rep = C.validate_answer(r["answer"], r["retrieved_ids"])
        for s in rep.sentences:
            if not s.ids:
                continue
            text = C.strip_citation_markers(s.text)
            for cid in s.ids:
                real_pool.append((r["qid"], text, cid))
            others = [i for i in r["retrieved_ids"] if i not in s.ids]
            if others:
                swap_pool.append((r["qid"], text, rng.choice(others)))
    n_swap = min(args.n_swapped, len(swap_pool), args.n)
    n_real = min(args.n - n_swap, len(real_pool))
    # round-robin over questions so one long answer cannot dominate the sample
    by_q = {}
    for it in real_pool:
        by_q.setdefault(it[0], []).append(it)
    for v in by_q.values():
        rng.shuffle(v)
    qs = sorted(by_q)
    rng.shuffle(qs)
    real = []
    while len(real) < n_real and any(by_q.values()):
        for q in qs:
            if by_q[q] and len(real) < n_real:
                real.append(by_q[q].pop())
    swapped = rng.sample(swap_pool, n_swap)
    rows = [(*it, "real") for it in real] + [(*it, "swapped") for it in swapped]
    rng.shuffle(rows)
    if len(rows) < args.n:
        print(f"WARNING: only {len(rows)} candidate pairs available (wanted {args.n}); "
              f"generate answers for more questions", file=sys.stderr)

    key = {}
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "sentence", "cited_id", "title", "abstract", "label"])
        for i, (qid, sentence, cid, kind) in enumerate(rows, 1):
            pid = f"c{i:02d}"
            key[pid] = {"qid": qid, "kind": kind}
            w.writerow([pid, sentence, cid, snap[cid]["title"], snap[cid]["summary"], ""])
    write_json(args.key, key)
    print(f"wrote {len(rows)} rows to {args.out} ({n_real} real, {n_swap} swapped). "
          f"Fill the `label` column (supported / partial / unsupported).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
