"""
Citation checker.  For every answered (non-refused, non-degraded) answer and every cited paper:

  1. in_retrieved  the cited ID is among the papers retrieved for that question
  2. resolves      the ID exists on arXiv (API id_list lookup; 'unknown' if arXiv is unreachable)
  3. supports      the cited abstract supports the sentence (NLI and/or LLM judge)

    python -m eval.check_citations --answers results/answers.jsonl \
        --snapshot eval/data/corpus_snapshot.jsonl --judge both --resolve

Support thresholds: NLI uses `nli_threshold` from results/calibration.json (written by
calibrate_judge.py).  Without it the checker falls back to 0.5 and marks the output
`calibrated: false`; treat such numbers as indicative only.
A sentence with several citations counts as supported if ANY cited abstract supports it
(a claim that needs two papers jointly is therefore under-counted).
Also reports first-pass compliance of the RAW model output (before repair/dropping).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from arxiv_research_companion import citations as C
from eval import metrics as M
from eval.common import RESULTS_DIR, load_snapshot, read_json, read_jsonl, run_meta, write_json, write_jsonl


def prop(k, n):
    p, lo, hi = M.wilson(k, n)
    return {"k": k, "n": n, "p": p, "ci95": [lo, hi]}


def summarize_pairs(pairs, nli_threshold=None):
    """pairs: dicts with qid, sentence_idx, cited_id, nli (float|None), llm (verdict|None)."""
    out = {"n_citation_pairs": len(pairs)}
    sents = {}
    for p in pairs:
        sents.setdefault((p["qid"], p["sentence_idx"]), []).append(p)
    out["n_cited_sentences"] = len(sents)
    if nli_threshold is not None and any(p.get("nli") is not None for p in pairs):
        ok = [p["nli"] >= nli_threshold for p in pairs if p.get("nli") is not None]
        out["nli_pair_support"] = prop(sum(ok), len(ok))
        s_ok = [any(q["nli"] >= nli_threshold for q in v) for v in sents.values()]
        out["nli_sentence_support"] = prop(sum(s_ok), len(s_ok))
    if any(p.get("llm") for p in pairs):
        ok = [p["llm"] == "supported" for p in pairs if p.get("llm")]
        out["llm_pair_support"] = prop(sum(ok), len(ok))
        s_ok = [any(q.get("llm") == "supported" for q in v) for v in sents.values()]
        out["llm_sentence_support"] = prop(sum(s_ok), len(s_ok))
        out["llm_verdicts"] = dict(Counter(p["llm"] for p in pairs if p.get("llm")))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", default=str(RESULTS_DIR / "answers.jsonl"))
    ap.add_argument("--snapshot", default="eval/data/corpus_snapshot.jsonl")
    ap.add_argument("--judge", choices=["nli", "llm", "both", "none"], default="both")
    ap.add_argument("--resolve", action="store_true", help="resolve cited IDs on arXiv (network)")
    ap.add_argument("--calibration", default=str(RESULTS_DIR / "calibration.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "citation_check.json"))
    args = ap.parse_args(argv)

    abstracts = {p["arxiv_id"]: p["summary"] for p in load_snapshot(args.snapshot)}
    records = [r for r in read_jsonl(args.answers) if r["kind"] == "in_scope"]
    answered = [r for r in records if not r["refused"] and not r["degraded"]]

    threshold, calibrated = 0.5, False
    if Path(args.calibration).exists():
        cal = read_json(args.calibration)
        if cal.get("nli_threshold") is not None:
            threshold, calibrated = cal["nli_threshold"], True

    # --- raw-output discipline and format ------------------------------------------------
    raw_viol = Counter()
    first_pass = 0
    n_raw = 0
    for r in answered:
        cit = r.get("citations") or {}
        first_pass += bool(cit.get("first_pass_ok"))
        if r.get("raw_answer"):
            n_raw += 1
            rep = C.validate_answer(r["raw_answer"], r["retrieved_ids"])
            for t in {v["type"] for v in rep.violations}:
                raw_viol[t] += 1
    fmt = {"n_answered": len(answered), "first_pass_compliant": prop(first_pass, len(answered)),
           "raw_answers_with_violation_type": dict(raw_viol), "n_raw_answers": n_raw,
           "answers_with_dropped_sentences": sum(bool((r.get("citations") or {}).get("dropped_sentences")) for r in answered),
           "n_dropped_sentences": sum(len((r.get("citations") or {}).get("dropped_sentences") or []) for r in answered)}

    # --- final answers: in_retrieved + pairs --------------------------------------------
    pairs, final_viol = [], Counter()
    n_final_sentences_needing = n_final_cited = 0
    for r in answered:
        rep = C.validate_answer(r["answer"], r["retrieved_ids"])
        for v in rep.violations:
            final_viol[v["type"]] += 1
        for i, s in enumerate(rep.sentences):
            if s.needs_citation:
                n_final_sentences_needing += 1
                n_final_cited += bool(s.valid_numbers)
            for cid in s.ids:
                pairs.append({"qid": r["qid"], "sentence_idx": i, "sentence": s.text, "cited_id": cid,
                              "in_retrieved": cid in r["retrieved_ids"], "nli": None, "llm": None})
    in_ret = prop(sum(p["in_retrieved"] for p in pairs), len(pairs))

    resolve = None
    if args.resolve:
        from eval.arxiv_api import resolve_ids
        ids = sorted({p["cited_id"] for p in pairs})
        status = resolve_ids(ids, cache_path=RESULTS_DIR / "cache" / "arxiv_resolve.json")
        for p in pairs:
            p["resolves"] = status[p["cited_id"]]
        c = Counter(status.values())
        resolve = {"unique_ids": len(ids), **dict(c),
                   "not_found_ids": [i for i, s in status.items() if s == "not_found"]}

    # --- support ---------------------------------------------------------------------------
    judges_used = []
    if args.judge in ("nli", "both") and pairs:
        from eval.judges import NliJudge
        j = NliJudge()
        judges_used.append(f"nli:{j.model_name}")
        for n, p in enumerate(pairs, 1):
            p["nli"] = j.score(abstracts[p["cited_id"]], p["sentence"])["score"]
    if args.judge in ("llm", "both") and pairs:
        from eval.judges import LlmJudge
        j = LlmJudge()
        judges_used.append(f"llm:{j.model_name}")
        for n, p in enumerate(pairs, 1):
            p["llm"] = j.score(abstracts[p["cited_id"]], p["sentence"])["verdict"]
            if n % 10 == 0:
                print(f"  llm-judged {n}/{len(pairs)}", flush=True)

    summary = summarize_pairs(pairs, threshold if args.judge in ("nli", "both") else None)
    result = {"meta": run_meta(answers=args.answers, judges=judges_used, nli_threshold=threshold,
                               calibrated=calibrated, n_in_scope=len(records)),
              "format": fmt,
              "final": {"sentences_needing_citation": n_final_sentences_needing,
                        "with_valid_citation": prop(n_final_cited, n_final_sentences_needing),
                        "violations_remaining": dict(final_viol),
                        "cited_id_in_retrieved": in_ret},
              "resolve": resolve, "support": summary,
              "n_degraded_excluded": sum(r["degraded"] for r in records),
              "n_refused_excluded": sum(r["refused"] for r in records)}
    write_json(args.out, result)
    write_jsonl(Path(args.out).with_suffix(".pairs.jsonl"), pairs)
    print(f"calibrated threshold: {calibrated}  (nli_threshold={threshold})")
    print({k: result[k] for k in ("format", "final", "resolve", "support")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
