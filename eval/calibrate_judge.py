"""
Calibrate the support judges on your hand labels.

    python -m eval.calibrate_judge --labels eval/data/calibration_to_label.csv \
        --snapshot eval/data/corpus_snapshot.jsonl --judge both

Binary target: human label == "supported" (partial counts as NOT supported: strict).
NLI judge   ROC-AUC; Youden-optimal threshold on all pairs; honest estimate via leave-one-out
            (threshold chosen on the other n-1 pairs) with Wilson CI.  Writes the threshold to
            results/calibration.json, which check_citations.py then uses.
LLM judge   accuracy / precision / recall / Cohen's kappa against the labels, plus the 3-class
            confusion.
With n = 30 every interval is wide (about +/-15 points); the script says so in its output.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from eval import metrics as M
from eval.common import RESULTS_DIR, load_snapshot, read_json, run_meta, write_json

LABELS = ("supported", "partial", "unsupported")


def candidate_thresholds(scores: Sequence[float]) -> List[float]:
    s = sorted(set(scores))
    mids = [(a + b) / 2 for a, b in zip(s, s[1:])]
    return [s[0] - 1e-9] + mids + [s[-1] + 1e-9]


def best_threshold(scores: Sequence[float], y: Sequence[int]) -> Dict:
    """Max Youden J = TPR - FPR; among ties take the median candidate (avoids an edge bias)."""
    P, N = sum(y), len(y) - sum(y)
    if P == 0 or N == 0:
        return {"threshold": None, "j": float("nan")}
    best, ties = -2.0, []
    for t in candidate_thresholds(scores):
        tp = sum(1 for s, l in zip(scores, y) if l == 1 and s >= t)
        fp = sum(1 for s, l in zip(scores, y) if l == 0 and s >= t)
        j = tp / P - fp / N
        if j > best + 1e-12:
            best, ties = j, [t]
        elif abs(j - best) <= 1e-12:
            ties.append(t)
    return {"threshold": ties[len(ties) // 2], "j": best}


def binary_report(pred: Sequence[int], y: Sequence[int]) -> Dict:
    tp = sum(1 for p, t in zip(pred, y) if p and t)
    fp = sum(1 for p, t in zip(pred, y) if p and not t)
    fn = sum(1 for p, t in zip(pred, y) if not p and t)
    tn = sum(1 for p, t in zip(pred, y) if not p and not t)
    acc = M.wilson(tp + tn, len(y))
    return {"n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc[0], "accuracy_ci95": [acc[1], acc[2]],
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "kappa": M.cohen_kappa(list(pred), list(y))}


def loo_eval(scores: Sequence[float], y: Sequence[int]) -> Dict:
    preds = []
    for i in range(len(y)):
        s_tr = [s for j, s in enumerate(scores) if j != i]
        y_tr = [t for j, t in enumerate(y) if j != i]
        thr = best_threshold(s_tr, y_tr)["threshold"]
        preds.append(int(thr is not None and scores[i] >= thr))
    return binary_report(preds, y)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="eval/data/calibration_to_label.csv")
    ap.add_argument("--snapshot", default="eval/data/corpus_snapshot.jsonl")
    ap.add_argument("--judge", choices=["nli", "llm", "both"], default="both")
    ap.add_argument("--out", default=str(RESULTS_DIR / "calibration.json"))
    ap.add_argument("--key", default=str(RESULTS_DIR / "calibration_key.json"))
    args = ap.parse_args(argv)

    with open(args.labels, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    bad = [r["pair_id"] for r in rows if (r.get("label") or "").strip().lower() not in LABELS]
    if bad:
        print(f"ERROR: missing/invalid labels for {bad[:10]} (allowed: {LABELS})", file=sys.stderr)
        return 1
    labels = [r["label"].strip().lower() for r in rows]
    y = [int(l == "supported") for l in labels]
    warnings = []
    if len(rows) < 30:
        warnings.append(f"only {len(rows)} labelled pairs (<30)")
    if min(sum(y), len(y) - sum(y)) < 8:
        warnings.append(f"class balance {sum(y)} supported / {len(y) - sum(y)} not: threshold estimate is unreliable")
    warnings.append(f"n={len(rows)}: all intervals are wide")
    abstracts = {p["arxiv_id"]: p["summary"] for p in load_snapshot(args.snapshot)}
    key_path = Path(args.key)
    kinds = [read_json(key_path).get(r["pair_id"], {}).get("kind") for r in rows] if key_path.exists() else None

    result = {"meta": run_meta(labels_file=args.labels), "n": len(rows),
              "label_counts": dict(Counter(labels)), "warnings": warnings, "nli_threshold": None}
    if kinds:
        result["kind_counts"] = dict(Counter(kinds))

    if args.judge in ("nli", "both"):
        from eval.judges import NliJudge
        j = NliJudge()
        scores = [j.score(abstracts[r["cited_id"]], r["sentence"])["score"] for r in rows]
        bt = best_threshold(scores, y)
        result["nli"] = {"model": j.model_name, "auc": M.auc(scores, y), "best_threshold_all_pairs": bt,
                         "at_best_threshold_in_sample": binary_report([int(s >= bt["threshold"]) for s in scores], y)
                         if bt["threshold"] is not None else None,
                         "leave_one_out": loo_eval(scores, y), "scores": scores}
        result["nli_threshold"] = bt["threshold"]
    if args.judge in ("llm", "both"):
        from eval.judges import LlmJudge
        j = LlmJudge()
        verdicts = []
        for n, r in enumerate(rows, 1):
            verdicts.append(j.score(abstracts[r["cited_id"]], r["sentence"])["verdict"])
        pred = [int(v == "supported") for v in verdicts]
        conf = Counter(zip(labels, verdicts))
        result["llm"] = {"model": j.model_name, "binary": binary_report(pred, y),
                         "unparsed": sum(v == "unparsed" for v in verdicts),
                         "confusion_human_vs_judge": {f"{h}->{v}": c for (h, v), c in sorted(conf.items())},
                         "kappa_3class": M.cohen_kappa(labels, verdicts)}
    write_json(args.out, result)
    print({k: v for k, v in result.items() if k not in ("meta",)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
