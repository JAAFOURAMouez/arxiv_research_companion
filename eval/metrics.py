"""
Metrics.

* ``ranx`` (``ranx_means``) produces the headline numbers.
* ``per_query`` is a small pure-Python reference implementation used (a) for paired
  statistics, which need per-query scores, and (b) to cross-check ranx: if the two disagree
  by more than 1e-6 on any metric the run aborts rather than reporting either.

Definitions (binary relevance):
  recall@k   = |relevant in top-k| / |relevant|     (can be < 1 at k=1 when a query has 2+ relevant)
  hit@k      = 1 if any relevant in top-k            (reference only)
  mrr@10     = 1/rank of first relevant within top 10, else 0
  ndcg@10    = DCG@10 / IDCG@10, gain 1, discount 1/log2(rank+1)
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KS = (1, 3, 5, 10)


def ranx_metric_names(ks: Sequence[int] = KS) -> List[str]:
    return [f"recall@{k}" for k in ks] + ["mrr@10", "ndcg@10"]


def report_metric_names(ks: Sequence[int] = KS) -> List[str]:
    return [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + ["mrr@10", "ndcg@10"]


RANX_METRICS = ranx_metric_names()
REPORT_METRICS = report_metric_names()
COST_ORDER = ["bm25", "dense", "hybrid", "hybrid_rerank"]  # cheapest -> most expensive


def per_query(ranking: Sequence[str], relevant: set, ks: Sequence[int] = KS) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n_rel = len(relevant)
    for k in ks:
        got = sum(1 for d in ranking[:k] if d in relevant)
        out[f"recall@{k}"] = got / n_rel if n_rel else 0.0
        out[f"hit@{k}"] = 1.0 if got else 0.0
    out["mrr@10"] = 0.0
    for i, d in enumerate(ranking[:10], start=1):
        if d in relevant:
            out["mrr@10"] = 1.0 / i
            break
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranking[:10], start=1) if d in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_rel, 10) + 1))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def score_rankings(rankings: Dict[str, List[str]], qrels: Dict[str, set],
                   ks: Sequence[int] = KS) -> Dict[str, Dict[str, float]]:
    return {qid: per_query(rankings[qid], qrels[qid], ks) for qid in qrels}


def column(pq: Dict[str, Dict[str, float]], metric: str, qids: Sequence[str]) -> List[float]:
    return [pq[q][metric] for q in qids]


def ranx_means(rankings: Dict[str, List[str]], qrels: Dict[str, set],
               metrics: Sequence[str] = RANX_METRICS) -> Dict[str, float]:
    from ranx import Qrels, Run, evaluate  # imported lazily: only the `eval` group has it

    q = Qrels.from_dict({qid: {d: 1 for d in rel} for qid, rel in qrels.items()})
    # strictly decreasing scores => rank order is exactly the order we pass in
    r = Run.from_dict({qid: {d: float(len(ranking) - i) for i, d in enumerate(ranking)}
                       for qid, ranking in rankings.items() if qid in qrels})
    res = evaluate(q, r, list(metrics))
    return {m: float(res[m]) for m in metrics} if isinstance(res, dict) else {metrics[0]: float(res)}


def mean_of(pq: Dict[str, Dict[str, float]], metric: str) -> float:
    return float(np.mean([v[metric] for v in pq.values()])) if pq else float("nan")


def evaluate_run(rankings: Dict[str, List[str]], qrels: Dict[str, set], engine: str = "ranx",
                 tol: float = 1e-6, ks: Sequence[int] = KS) -> dict:
    """Means (ranx or reference), reference per-query scores, and the ranx/reference agreement."""
    pq = score_rankings(rankings, qrels, ks)
    ref_means = {m: mean_of(pq, m) for m in report_metric_names(ks)}
    out = {"per_query": pq, "reference_means": ref_means, "engine": engine}
    if engine == "ranx":
        rx = ranx_means(rankings, qrels, ranx_metric_names(ks))
        diff = max(abs(rx[m] - ref_means[m]) for m in rx)
        if diff > tol:
            raise RuntimeError(
                f"ranx and the reference implementation disagree (max |diff| = {diff:.2e}); "
                f"refusing to report. ranx={rx} reference={ {m: ref_means[m] for m in rx} }")
        out["means"] = {**ref_means, **rx}
        out["ranx_vs_reference_max_abs_diff"] = diff
    else:
        out["means"] = ref_means
        out["ranx_vs_reference_max_abs_diff"] = None
    return out


# --------------------------------------------------------------------------- #
# uncertainty                                                                 #
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: Sequence[float], n: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n: int = 10000,
                          alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Mean(a-b), percentile CI of the paired difference and a two-sided bootstrap p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1)
    p = 2 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {"mean_diff": float(d.mean()), "ci_lo": float(np.quantile(means, alpha / 2)),
            "ci_hi": float(np.quantile(means, 1 - alpha / 2)), "p_two_sided": min(1.0, p), "alpha": alpha}


def random_baseline(doc_ids: Sequence[str], qrels: Dict[str, set], n_perm: int = 200,
                    seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Expected per-query metrics of a uniformly random ranking (mean over n_perm shuffles)."""
    rng = random.Random(seed)
    ids = list(doc_ids)
    acc: Dict[str, Dict[str, float]] = {q: {} for q in qrels}
    for _ in range(n_perm):
        rng.shuffle(ids)
        top = ids[:10]
        for q, rel in qrels.items():
            for m, v in per_query(top, rel).items():
                acc[q][m] = acc[q].get(m, 0.0) + v / n_perm
    return acc


# --------------------------------------------------------------------------- #
# pre-declared configuration choice                                           #
# --------------------------------------------------------------------------- #
def select_config(scores: Dict[str, List[float]], *, metric: str = "ndcg@10", alpha: float = 0.05,
                  order: Sequence[str] = COST_ORDER, n_boot: int = 10000, seed: int = 0) -> dict:
    """Choose the cheapest configuration that is not distinguishable from the best.

    Rule (fixed before looking at any number):
      1. ``best`` = highest mean of ``metric`` over queries.
      2. For every other configuration c, take the paired bootstrap CI of mean(best - c) with
         Bonferroni-adjusted level alpha/(m-1).  If the CI contains 0, c is *indistinguishable*
         from best.
      3. Chosen = the cheapest (order: bm25 < dense < hybrid < hybrid_rerank) among best and
         the indistinguishable ones.
    Consequence worth knowing: with few queries most differences are not significant, so the
    rule then returns the cheapest configuration.  That is a statement about the data's power,
    not a finding that the cheap method is as good.
    """
    names = [c for c in order if c in scores]
    means = {c: float(np.mean(scores[c])) for c in names}
    best = max(names, key=lambda c: (means[c], -names.index(c)))  # ties -> cheaper
    m = len(names)
    alpha_adj = alpha / max(1, m - 1)
    comparisons, candidates = {}, []
    for c in names:
        if c == best:
            candidates.append(c)
            continue
        cmp = paired_bootstrap_diff(scores[best], scores[c], n=n_boot, alpha=alpha_adj, seed=seed)
        cmp["indistinguishable"] = cmp["ci_lo"] <= 0.0 <= cmp["ci_hi"]
        comparisons[c] = cmp
        if cmp["indistinguishable"]:
            candidates.append(c)
    chosen = min(candidates, key=names.index)
    return {"metric": metric, "n_queries": len(next(iter(scores.values()))) if scores else 0,
            "means": means, "best_by_mean": best, "chosen": chosen,
            "alpha_adjusted": alpha_adj, "comparisons_vs_best": comparisons,
            "rule": "cheapest config whose paired-bootstrap CI vs the best includes 0 "
                    "(Bonferroni-adjusted); cost order " + " < ".join(order)}


# --------------------------------------------------------------------------- #
# proportions / agreement                                                     #
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """(p, lo, hi) Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == l for x in a) / n) * (sum(y == l for y in b) / n) for l in labels)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def auc(scores: Sequence[float], y: Sequence[int]) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties count 1/2)."""
    pos = [s for s, t in zip(scores, y) if t == 1]
    neg = [s for s, t in zip(scores, y) if t == 0]
    if not pos or not neg:
        return float("nan")
    u = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return u / (len(pos) * len(neg))
