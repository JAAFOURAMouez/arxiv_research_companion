"""
Assemble results/RESULTS.md from the JSON files the other scripts wrote.

    python -m eval.report

Only values found in the JSON files are printed; anything not run shows 'NOT RUN'.  Runs made with
stand-ins (hash embedder, fake reranker, reference engine) are stamped SMOKE and must not be
quoted.  The file ends with the standing list of what this pipeline cannot verify.
"""
from __future__ import annotations

import sys
from pathlib import Path

from eval.common import RESULTS_DIR, read_json

NR = "NOT RUN"


def load(name):
    p = RESULTS_DIR / name
    return read_json(p) if p.exists() else None


def f(x, nd=3):
    return NR if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def prop(d):
    return NR if not d else f"{d['p']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] (n={d['n']})"


def smoke_banner(meta):
    return "> **SMOKE RUN (stand-ins used) - NOT RESULTS**\n\n" if meta and meta.get("smoke") else ""


def retrieval_section(r):
    if not r:
        return f"## 2. Retrieval\n\n{NR}\n"
    out = ["## 2. Retrieval\n", smoke_banner(r["meta"]),
           f"Embedder `{r['meta']['embedder']}`, dense backend `{r['meta']['dense_backend']}`, doc text "
           f"`{r['meta']['doc_text']}`, engine `{r['meta']['engine']}`, snapshot sha256 "
           f"`{(r['meta'].get('snapshot_sha256') or 'n/a')[:12]}`, cross-encoder `{r['meta'].get('cross_encoder')}`.\n"]
    for mode, m in r["modes"].items():
        tag = " (primary)" if mode == r["primary_mode"] else ""
        out.append(f"### Corpus mode `{mode}`{tag}: {m['n_docs']} docs, {m['n_queries']} queries "
                   f"({len(m['dropped_queries_no_relevant_in_corpus'])} dropped: no relevant paper in corpus; "
                   f"{m['labeled_ids_in_corpus']}/{m['labeled_ids']} labelled IDs present)\n")
        if "error" in m:
            out.append(m["error"] + "\n")
            continue
        cols = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
        out.append("| config | " + " | ".join(cols) + " | p50 ms | p95 ms |")
        out.append("|---" * (len(cols) + 3) + "|")
        for n, c in m["configs"].items():
            cells = [f"{c['means'][k]:.3f} [{c['ci95'][k][0]:.2f}, {c['ci95'][k][1]:.2f}]" for k in cols]
            out.append(f"| {n} | " + " | ".join(cells) + f" | {c['latency']['p50_ms']} | {c['latency']['p95_ms']} |")
        rb = m["random_baseline"]["means"]
        out.append("| random (expected) | " + " | ".join(f"{rb[k]:.3f}" for k in cols) + " | - | - |")
        out.append("\nhit@k (any relevant in top k): " + "; ".join(
            f"{n}: " + "/".join(f"{c['means'][f'hit@{k}']:.2f}" for k in (1, 3, 5, 10)) for n, c in m["configs"].items()))
        s = m["selection"]
        out.append(f"\n**Pre-declared selection rule** ({s['rule']}): best by mean nDCG@10 = `{s['best_by_mean']}`, "
                   f"chosen = `{s['chosen']}`.")
        out.append("\n| config | mean nDCG@10 | best - config, CI (adj.) | indistinguishable from best |")
        out.append("|---|---|---|---|")
        for n, mean in s["means"].items():
            cmp = s["comparisons_vs_best"].get(n)
            out.append(f"| {n} | {mean:.3f} | " + (f"{cmp['mean_diff']:+.3f} [{cmp['ci_lo']:+.3f}, {cmp['ci_hi']:+.3f}]" if cmp else "(best)")
                       + f" | {('yes' if cmp['indistinguishable'] else 'no') if cmp else '-'} |")
        zero = {n: c["recall@10_zero_queries"] for n, c in m["configs"].items()}
        out.append("\nQueries with recall@10 = 0: " + "; ".join(f"{n}: {len(v)} {v}" for n, v in zero.items()) + "\n")
    return "\n".join(out) + "\n"


def agree_section(a):
    if not a:
        return f"## 2b. ONNX vs sentence-transformers\n\n{NR}\n"
    d, q = a["docs"], a.get("queries")
    out = ["## 2b. ONNX vs sentence-transformers\n", smoke_banner(a["meta"]),
           "| text set | n | min cos | p01 | median | frac >= thr | # below thr |", "|---|---|---|---|---|---|---|",
           f"| abstracts | {d['n']} | {d['min']:.5f} | {d['p01']:.5f} | {d['median']:.5f} | {d['frac_ge_threshold']:.4f} | {d['n_below_threshold']} |"]
    if q:
        out.append(f"| questions | {q['n']} | {q['min']:.5f} | {q['p01']:.5f} | {q['median']:.5f} | {q['frac_ge_threshold']:.4f} | {q['n_below_threshold']} |")
        out.append(f"\nSame top-1 neighbour for {q['top1_same_fraction']:.3f} of questions; mean top-10 overlap {q['top10_overlap_mean']:.3f}.")
    out.append(f"\nThreshold {a['meta']['threshold']}: **{'PASS' if a['pass'] else 'FAIL'}**. Worst abstracts: {d['worst5']}\n")
    return "\n".join(out) + "\n"


def scifact_section(s):
    if not s:
        return f"## 1b. BEIR SciFact cross-check\n\n{NR}\n"
    m = s["meta"]
    out = ["## 1b. BEIR SciFact cross-check\n", smoke_banner(m),
           f"{m['n_docs']} docs, {m['n_queries']} queries, full test split: **{m['full_test_split']}**.\n",
           "| row | nDCG@10 [95% CI] | recall@10 | recall@100 | MRR@10 |", "|---|---|---|---|---|"]
    for n, r in s["rows"].items():
        mm = r["means"]
        out.append(f"| {n} | {mm['ndcg@10']:.4f} [{r['ndcg@10_ci95'][0]:.3f}, {r['ndcg@10_ci95'][1]:.3f}] | {mm['recall@10']:.4f} | {mm['recall@100']:.4f} | {mm['mrr@10']:.4f} |")
    out.append("\n| embedder | mine | published | diff | status |\n|---|---|---|---|---|")
    for k, c in s["published_check"].items():
        out.append(f"| {k} | {f(c.get('mine'), 4)} | {f(c.get('published'), 4)} | {f(c.get('diff'), 4)} | {c['status']} |")
    if s.get("onnx_vs_st"):
        d = s["onnx_vs_st"]["onnx_minus_st_ndcg@10"]
        out.append(f"\nONNX - ST nDCG@10 (paired bootstrap): {d['mean_diff']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]")
    return "\n".join(out) + "\n"


def generation_section(cc, cal, rg, rf, meta):
    out = ["## 3. Generation\n"]
    if meta:
        out.append(smoke_banner(meta))
        out.append(f"Generator `{meta['llm_model']}` ({meta['llm_provider']}), retrieval `{meta['retrieval_mode']}`, "
                   f"corpus `{meta['corpus_mode']}`, k={meta['k']}, T={meta['temperature']}, enforcement={meta['enforce']}.\n")
    out.append("### Citation checks\n")
    if not cc:
        out.append(NR + "\n")
    else:
        fm, fi, sp = cc["format"], cc["final"], cc["support"]
        out += [f"Answered: {fm['n_answered']} (excluded: {cc['n_refused_excluded']} refused, {cc['n_degraded_excluded']} LLM failures).",
                f"- first-pass compliance of the raw model output: {prop(fm['first_pass_compliant'])}",
                f"- raw outputs with each violation type: {fm['raw_answers_with_violation_type']}",
                f"- sentences dropped by enforcement: {fm['n_dropped_sentences']} in {fm['answers_with_dropped_sentences']} answers",
                f"- final answer, factual sentences with a valid citation: {prop(fi['with_valid_citation'])}",
                f"- final answer, cited IDs inside the retrieved set: {prop(fi['cited_id_in_retrieved'])}; violations left: {fi['violations_remaining']}",
                f"- IDs resolving on arXiv: {cc['resolve'] if cc['resolve'] else NR}",
                f"- judges: {cc['meta']['judges']}; NLI threshold {cc['meta']['nli_threshold']} (**calibrated: {cc['meta']['calibrated']}**)"]
        for k in ("nli_pair_support", "nli_sentence_support", "llm_pair_support", "llm_sentence_support"):
            out.append(f"- {k}: {prop(sp.get(k))}")
        out.append("")
    out.append("### Judge calibration (hand-labelled pairs)\n")
    if not cal:
        out.append(NR + "\n")
    else:
        out.append(f"n={cal['n']}, labels {cal['label_counts']}, kinds {cal.get('kind_counts')}. Warnings: {cal['warnings']}")
        if "nli" in cal:
            n = cal["nli"]
            out.append(f"- NLI `{n['model']}`: AUC {f(n['auc'])}, threshold {f(n['best_threshold_all_pairs']['threshold'])}; "
                       f"leave-one-out accuracy {prop({'p': n['leave_one_out']['accuracy'], 'ci95': n['leave_one_out']['accuracy_ci95'], 'n': n['leave_one_out']['n']})}, "
                       f"precision {f(n['leave_one_out']['precision'])}, recall {f(n['leave_one_out']['recall'])}, kappa {f(n['leave_one_out']['kappa'])}")
        if "llm" in cal:
            l = cal["llm"]
            b = l["binary"]
            out.append(f"- LLM judge `{l['model']}`: accuracy {prop({'p': b['accuracy'], 'ci95': b['accuracy_ci95'], 'n': b['n']})}, "
                       f"precision {f(b['precision'])}, recall {f(b['recall'])}, kappa {f(b['kappa'])}, "
                       f"3-class kappa {f(l['kappa_3class'])}, unparsed {l['unparsed']}; confusion {l['confusion_human_vs_judge']}")
        out.append("")
    out.append("### RAGAS faithfulness\n")
    out.append(NR + "\n" if not rg else
               f"{rg['faithfulness_mean']:.3f} [{rg['ci95'][0]:.3f}, {rg['ci95'][1]:.3f}] over {rg['n_scored']} answers "
               f"({rg['n_nan']} NaN), judge `{rg['meta']['judge_model']}`\n")
    out.append("### Out-of-corpus refusal\n")
    if not rf:
        out.append(NR + "\n")
    else:
        out += [f"- OOC refused: {prop(rf['ooc_refusal'])}; answered instead: {rf['ooc_answered_instead_of_refusing']}",
                f"- in-scope falsely refused: {prop(rf['in_scope_false_refusal'])}; {rf['in_scope_false_refusal_qids']}",
                f"- LLM failures excluded: {rf['llm_failures_excluded']}; unaudited OOC excluded: {rf['n_ooc_unaudited_excluded']}",
                f"- top-1 cosine AUROC in-scope vs OOC: {f(rf['top1_cosine_auroc_inscope_vs_ooc'])} "
                f"(in-scope {rf['top1_cosine_in_scope']}, OOC {rf['top1_cosine_ooc']})\n"]
    return "\n".join(out) + "\n"


UNVERIFIED = """## 5. What remains unverified (standing list)

- Every number above is only as good as the 40 labels: one labeller, no inter-annotator agreement, relevance judged from arXiv.org rather than pooled from system outputs (so relevant papers nobody labelled count as misses).
- With ~40 queries the confidence intervals are wide; differences between configurations are usually not significant, and the selection rule then returns the cheapest one.
- The corpus is a few hundred recent abstracts; recall@10 over so few documents is easy. Compare against the random row.
- SciFact is claim verification over biomedical abstracts, not question answering over arXiv cs.*; it validates the embedding pipeline, not the product's retrieval quality.
- The published reference value was entered by you; the tolerance is a judgment call.
- Support judges are imperfect proxies; calibration uses 30 pairs by one labeller; the sentence-level 'any cited paper supports it' rule under-counts multi-paper claims.
- Abstract-only grounding: a sentence can be true of the full paper yet unsupported by the abstract, and is counted unsupported.
- Generation results depend on the Groq model version on the day, sampling temperature and rate-limit behaviour; one run is one sample.
- RAGAS API usage, the BEIR download URL, the default Groq judge model name and Streamlit AppTest accessors were written without being executed.
- Peak RSS is for the evaluation process, not the Streamlit Cloud container.
"""


def main() -> int:
    ans_meta = RESULTS_DIR / "answers.meta.json"
    gm = read_json(ans_meta) if ans_meta.exists() else None
    parts = ["# Results\n",
             "Generated by `python -m eval.report` from the JSON files in `results/`. Nothing is typed by hand.\n",
             scifact_section(load("scifact.json")), retrieval_section(load("retrieval.json")),
             agree_section(load("embeddings_agree.json")),
             generation_section(load("citation_check.json"), load("calibration.json"),
                                load("ragas.json"), load("refusal.json"), gm),
             "## 4. Reliability\n\nSee `pytest` / GitHub Actions status; the nightly workflow's job summary lists each ingestion.\n",
             UNVERIFIED]
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {RESULTS_DIR / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
