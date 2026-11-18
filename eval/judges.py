"""
Support judges: does the cited abstract support a generated sentence?

NliJudge   cross-encoder NLI (default cross-encoder/nli-deberta-v3-base, DeBERTa family).
           The abstract is split into overlapping 3-sentence windows (+ the whole abstract) and
           the MAX entailment probability is kept.  Max-over-windows raises recall and also the
           false-positive rate: that is exactly what the 30-sentence calibration is for.
LlmJudge   a chat model from a DIFFERENT family than the generator (generator = OpenAI gpt-oss
           on Groq; default judge = Meta Llama on Groq, override with JUDGE_MODEL).

Neither is ground truth: both are scored against your hand labels in calibrate_judge.py.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional

from arxiv_research_companion.citations import split_sentences, strip_citation_markers

DEFAULT_NLI = "cross-encoder/nli-deberta-v3-base"
DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"  # unverified default: check it still exists on Groq
# Judge defaults per provider. Ollama: a Qwen model, a different family from the default
# llama3.2 generator.
JUDGE_MODELS = {"groq": DEFAULT_JUDGE_MODEL, "ollama": "qwen2.5:7b", "openai": "gpt-4o-mini",
                "openrouter": "meta-llama/llama-3.3-70b-instruct"}


def judge_provider() -> str:
    """JUDGE_PROVIDER, else LLM_PROVIDER, else groq (the historical default)."""
    provider = (os.getenv("JUDGE_PROVIDER") or os.getenv("LLM_PROVIDER") or "groq").lower()
    return provider if provider in JUDGE_MODELS else "groq"


def default_judge_model(provider: Optional[str] = None) -> str:
    return os.getenv("JUDGE_MODEL") or JUDGE_MODELS[provider or judge_provider()]


def judge_api_key(provider: Optional[str] = None) -> Optional[str]:
    """Key from the provider's env var; Ollama needs none (a placeholder keeps clients happy)."""
    from arxiv_research_companion.llm_http import PROVIDERS

    spec = PROVIDERS[provider or judge_provider()]
    return os.getenv(spec.key_env) if spec.needs_key else "ollama"


def premise_windows(abstract: str, size: int = 3) -> List[str]:
    sents = split_sentences(abstract)
    wins = [" ".join(sents[i:i + size]) for i in range(max(1, len(sents) - size + 1))] if sents else []
    whole = abstract.strip()
    if whole and whole not in wins:
        wins.append(whole)
    return wins or [whole]


class NliJudge:
    name = "nli"

    def __init__(self, model_name: str = DEFAULT_NLI, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        ent = [i for i, l in labels.items() if l.startswith("entail")]
        if len(ent) != 1:
            raise RuntimeError(f"cannot find the entailment label in {labels}")
        self.ent_idx = ent[0]

    def entail_prob(self, premises: List[str], hypothesis: str) -> List[float]:
        enc = self.tok(premises, [hypothesis] * len(premises), truncation="only_first",
                       max_length=self.max_length, padding=True, return_tensors="pt")
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[:, self.ent_idx]
        return [float(x) for x in probs]

    def score(self, abstract: str, sentence: str) -> Dict:
        hyp = strip_citation_markers(sentence)
        wins = premise_windows(abstract)
        probs = self.entail_prob(wins, hyp)
        return {"score": max(probs), "n_windows": len(wins)}


_VERDICT = re.compile(r'"?verdict"?\s*[:=]\s*"?(supported|partial|unsupported)', re.I)
# qwen2.5:7b sometimes drops the key and answers {"supported"}; take the first bare verdict word.
_BARE_VERDICT = re.compile(r"\b(unsupported|supported|partial)\b", re.I)


def parse_verdict(raw: str) -> Optional[str]:
    m = _VERDICT.search(raw or "") or _BARE_VERDICT.search(raw or "")
    return m.group(1).lower() if m else None

JUDGE_PROMPT = """You are a strict fact-checker. Decide whether the ABSTRACT supports the CLAIM.

ABSTRACT:
{abstract}

CLAIM:
{claim}

Answer with one JSON object and nothing else: {{"verdict": "supported" | "partial" | "unsupported"}}
- supported: the abstract states or directly implies everything the claim says.
- partial: the abstract supports only part of the claim, or the claim overstates it.
- unsupported: the abstract does not support the claim, or contradicts it."""


class LlmJudge:
    name = "llm"
    SCORES = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 sleep_s: Optional[float] = None, provider: Optional[str] = None):
        self.provider = provider or judge_provider()
        self.model_name = model or default_judge_model(self.provider)
        self.api_key = api_key or judge_api_key(self.provider)
        # no rate limit to respect locally
        self.sleep_s = sleep_s if sleep_s is not None else (0.0 if self.provider == "ollama" else 1.0)
        if not self.api_key:
            from arxiv_research_companion.llm_http import PROVIDERS

            raise RuntimeError(f"{PROVIDERS[self.provider].key_env} is required for the LLM judge")

    def score(self, abstract: str, sentence: str) -> Dict:
        from arxiv_research_companion.llm_http import chat_completion

        prompt = JUDGE_PROMPT.format(abstract=abstract.strip(), claim=strip_citation_markers(sentence))
        time.sleep(self.sleep_s)
        raw = chat_completion([{"role": "user", "content": prompt}], model=self.model_name,
                              api_key=self.api_key, max_tokens=60, temperature=0.0,
                              timeout=180.0 if self.provider == "ollama" else 60.0,
                              provider=self.provider) or ""
        verdict = parse_verdict(raw) or "unparsed"
        return {"verdict": verdict, "score": self.SCORES.get(verdict, float("nan")), "raw": raw[:200]}
