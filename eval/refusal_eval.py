"""
Out-of-corpus refusal.

    python -m eval.refusal_eval --answers results/answers.jsonl

Needs an answers file produced with --ooc.  Reports, with Wilson CIs:
  refusal rate on audited OOC questions            (want high)
  false-refusal rate on in-scope questions         (want low; an always-refuse bot scores 100% on the first)
  LLM failures (degraded) are excluded from both and counted separately
  AUROC of the top-1 dense cosine for separating in-scope from OOC questions: tells you whether a
  retrieval-score gate could add anything beyond the model's own refusal (no gate is implemented).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from eval import metrics as M
from eval.common import RESULTS_DIR, read_jsonl, run_meta, write_json


def rate(k, n):
    p, lo, hi = M.wilson(k, n)
    return {"k": k, "n": n, "p": p, "ci95": [lo, hi]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", default=str(RESULTS_DIR / "answers.jsonl"))
    ap.add_argument("--include-unaudited", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR / "refusal.json"))
    args = ap.parse_args(argv)

    recs = read_jsonl(args.answers)
    ins = [r for r in recs if r["kind"] == "in_scope"]
    ooc_all = [r for r in recs if r["kind"] == "ooc"]
    aud = lambda r: str(r.get("audited", "")).lower() in ("1", "true", "yes")  # noqa: E731
    ooc = ooc_all if args.include_unaudited else [r for r in ooc_all if aud(r)]
    if not ooc:
        print("ERROR: no (audited) OOC records; run generate_answers with --ooc and audit the questions", file=sys.stderr)
        return 1
    ooc_ok = [r for r in ooc if not r["degraded"] or r["refused"]]
    ins_ok = [r for r in ins if not r["degraded"] or r["refused"]]
    answered_ooc = [r["qid"] for r in ooc_ok if not r["refused"]]
    top_in = [r["top1_dense_cosine"] for r in ins if r["top1_dense_cosine"] is not None]
    top_ooc = [r["top1_dense_cosine"] for r in ooc if r["top1_dense_cosine"] is not None]
    y = [1] * len(top_in) + [0] * len(top_ooc)
    result = {"meta": run_meta(answers=args.answers),
              "ooc_refusal": rate(sum(r["refused"] for r in ooc_ok), len(ooc_ok)),
              "ooc_answered_instead_of_refusing": answered_ooc,
              "in_scope_false_refusal": rate(sum(r["refused"] for r in ins_ok), len(ins_ok)),
              "in_scope_false_refusal_qids": [r["qid"] for r in ins_ok if r["refused"]],
              "llm_failures_excluded": {"ooc": len(ooc) - len(ooc_ok), "in_scope": len(ins) - len(ins_ok)},
              "n_ooc_unaudited_excluded": len(ooc_all) - len(ooc),
              "top1_cosine_auroc_inscope_vs_ooc": M.auc(top_in + top_ooc, y),
              "top1_cosine_in_scope": {"min": float(np.min(top_in)), "median": float(np.median(top_in)), "max": float(np.max(top_in))} if top_in else None,
              "top1_cosine_ooc": {"min": float(np.min(top_ooc)), "median": float(np.median(top_ooc)), "max": float(np.max(top_ooc))} if top_ooc else None}
    write_json(args.out, result)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
