# Results (template: blanks are for values produced by `python -m eval.report`)

Run date: ____ | git sha: ____ | snapshot sha256: ____ | Python 3.12 | uv.lock hash: ____
Generator: ____ | judge model: ____ | embedder: ____ | cross-encoder: ____

## 1b. BEIR SciFact cross-check (full test split? yes / no)
| row | nDCG@10 [95% CI] | recall@10 | recall@100 | MRR@10 |
|---|---|---|---|---|
| dense[onnx] | | | | |
| dense[st] | | | | |
| bm25[onnx] | | | | |
| hybrid[onnx] | | | | |
| hybrid_rerank[onnx] | | | | |

| embedder | mine | published (source URL) | diff | tolerance | status |
|---|---|---|---|---|---|
| onnx | | | | | |
| st | | | | | |

ONNX - ST nDCG@10, paired bootstrap: ____ [____, ____]

## 2. Retrieval (corpus mode: prod / augmented; docs ____; queries ____ of 40; labelled IDs present ____/____)
| config | recall@1 | recall@3 | recall@5 | recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms | peak RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| dense (ONNX MiniLM) | | | | | | | | | |
| bm25 (FTS5) | | | | | | | | | |
| hybrid (RRF) | | | | | | | | | |
| hybrid + cross-encoder | | | | | | | | | |
| random (expected) | | | | | | | - | - | - |

Selection rule output: best by mean = ____; chosen = ____; indistinguishable from best: ____
Queries with recall@10 = 0 per config: ____

## 2b. ONNX vs sentence-transformers (cosine, threshold 0.99)
| set | n | min | p01 | median | fraction >= 0.99 | PASS/FAIL |
|---|---|---|---|---|---|---|
| abstracts | | | | | | |
| questions | | | | | | |
Top-1 neighbour identical for ____ of questions; mean top-10 overlap ____

## 3. Generation
| measure | value [95% CI] | n |
|---|---|---|
| first-pass compliance of raw output | | |
| factual sentences with a valid [n] (final) | | |
| cited IDs inside retrieved set | | |
| cited IDs resolving on arXiv | | |
| sentence supported, NLI (calibrated? yes/no) | | |
| sentence supported, LLM judge | | |
| RAGAS faithfulness (judge ____) | | |
| OOC questions refused (audited only) | | |
| in-scope questions falsely refused | | |
| top-1 cosine AUROC, in-scope vs OOC | | |

Judge calibration (n=30): NLI AUC ____, threshold ____, LOO accuracy ____ [____, ____]; LLM judge accuracy ____, kappa ____

## 4. Reliability
pytest + AppTest on GitHub Actions: pass / fail (link) | nightly ingestion last run: ____ | uv.lock committed: yes / no

## 5. Negative results / surprises
(____)

## 6. Still unverified
(standing list is appended by the generator)
