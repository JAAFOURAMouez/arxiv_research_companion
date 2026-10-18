"""
Citation parsing, validation and enforcement for generated answers.

Contract given to the LLM (see ``rag_engine._construct_cited_prompt``):

* the context holds ``k`` numbered papers ``[1]`` .. ``[k]``;
* every sentence that states a fact ends with one or more ``[n]`` markers
  (before the final period);
* if the context cannot answer the question the model answers with the sentinel
  ``NOT_ENOUGH_INFORMATION`` and nothing else.

This module is pure Python (no third-party imports) so it is cheap to test.
Checking that a citation *supports* its sentence is NOT done here: that needs an NLI
model or a judge and lives in ``eval/check_citations.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

REFUSAL_SENTINEL = "NOT_ENOUGH_INFORMATION"

# [1]  [1,2]  [1; 3]  [2-4]  (en dash accepted)
_CITE_RE = re.compile(r"\[\s*(\d+(?:\s*(?:,|;|-|–)\s*\d+)*)\s*\]")
_NEW_ID_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")
_OLD_ID_RE = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b")

_ABBREVIATIONS = (
    "e.g.", "i.e.", "et al.", "vs.", "Fig.", "Figs.", "Eq.", "Eqs.", "Sec.",
    "cf.", "approx.", "resp.", "No.", "Dr.", "Prof.",
)
_DOT = "\x00"
_BOUNDARY = re.compile(
    r"""([.!?]+["'”’)]*                                  # terminal punctuation
        (?:\s*\[\s*\d+(?:\s*(?:,|;|-|–)\s*\d+)*\s*\])*)  # trailing [n] markers
        (\s+)(?=[A-Z0-9"“(\[*_])                         # whitespace, then a new sentence
    """,
    re.VERBOSE,
)
_BULLET_RE = re.compile(r"^(\s*(?:[-*•]\s+|\d+[.)]\s+))")


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #
def parse_marker(group: str) -> List[int]:
    """``"1, 3-5"`` -> ``[1, 3, 4, 5]`` (ranges capped at 20 members)."""
    numbers: List[int] = []
    for part in re.split(r"\s*[,;]\s*", group.strip()):
        m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            numbers.extend(range(lo, hi + 1) if lo <= hi and hi - lo <= 20 else [lo, hi])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def extract_numbers(text: str) -> List[int]:
    """All cited numbers in order of appearance (duplicates removed)."""
    seen: List[int] = []
    for m in _CITE_RE.finditer(text):
        for n in parse_marker(m.group(1)):
            if n not in seen:
                seen.append(n)
    return seen


def strip_citation_markers(text: str) -> str:
    """Remove ``[n]`` markers and tidy whitespace (used before RAGAS / NLI)."""
    out = _CITE_RE.sub("", text)
    out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def split_sentences(text: str) -> List[str]:
    """Sentence split that keeps ``[n]`` markers attached to their sentence."""
    sentences: List[str] = []
    for line in re.split(r"\n+", text):
        line = _BULLET_RE.sub("", line).strip()
        if not line:
            continue
        sentences.extend(_split_line(line))
    return sentences


def _protect(line: str) -> str:
    for abbr in _ABBREVIATIONS:
        pattern = r"(?<![A-Za-z])" + re.escape(abbr)
        line = re.sub(
            pattern, lambda m: m.group(0).replace(".", _DOT), line, flags=re.IGNORECASE
        )
    return line


def _split_line(line: str) -> List[str]:
    protected = _protect(line)
    parts: List[str] = []
    start = 0
    for m in _BOUNDARY.finditer(protected):
        parts.append(protected[start : m.end(1)])
        start = m.end()
    parts.append(protected[start:])
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


def is_refusal(text: str) -> bool:
    return REFUSAL_SENTINEL in (text or "")[:160]


def _word_count(sentence: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", strip_citation_markers(sentence)))


def needs_citation(sentence: str, min_words: int = 5) -> bool:
    """Short fragments (headings, 'In summary:') are not treated as factual claims."""
    return _word_count(sentence) >= min_words


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class SentenceCheck:
    text: str
    numbers: List[int]
    valid_numbers: List[int]
    invalid_numbers: List[int]
    needs_citation: bool
    ids: List[str]  # arXiv IDs of valid_numbers


@dataclass
class CitationReport:
    sentences: List[SentenceCheck] = field(default_factory=list)
    violations: List[Dict] = field(default_factory=list)
    refused: bool = False
    k: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def cited_ids(self) -> List[str]:
        out: List[str] = []
        for s in self.sentences:
            for i in s.ids:
                if i not in out:
                    out.append(i)
        return out

    def summary(self) -> Dict:
        needing = [s for s in self.sentences if s.needs_citation]
        return {
            "ok": self.ok,
            "refused": self.refused,
            "n_sentences": len(self.sentences),
            "n_needing_citation": len(needing),
            "n_with_valid_citation": sum(1 for s in needing if s.valid_numbers),
            "n_violations": len(self.violations),
            "violation_types": sorted({v["type"] for v in self.violations}),
            "cited_ids": self.cited_ids,
        }


def validate_answer(
    answer: str, source_ids: Sequence[str], min_words: int = 5
) -> CitationReport:
    """Check an answer against the ``[n]`` contract.

    Violation types: ``uncited`` (factual sentence without marker), ``out_of_range``
    (marker outside 1..k), ``unknown_arxiv_id`` (an arXiv ID typed in the prose that is
    not among the retrieved papers).
    """
    k = len(source_ids)
    report = CitationReport(k=k)
    if is_refusal(answer):
        report.refused = True
        return report

    for idx, sentence in enumerate(split_sentences(answer)):
        numbers = extract_numbers(sentence)
        valid = [n for n in numbers if 1 <= n <= k]
        invalid = [n for n in numbers if not 1 <= n <= k]
        needs = needs_citation(sentence, min_words)
        report.sentences.append(
            SentenceCheck(sentence, numbers, valid, invalid, needs, [source_ids[n - 1] for n in valid])
        )
        if invalid:
            report.violations.append(
                {"type": "out_of_range", "sentence": idx, "detail": f"cited {invalid}, valid range is 1-{k}"}
            )
        if needs and not valid:
            report.violations.append(
                {"type": "uncited", "sentence": idx, "detail": sentence[:120]}
            )
        known = {s.split("v")[0] for s in source_ids}
        for pattern in (_NEW_ID_RE, _OLD_ID_RE):
            for m in pattern.finditer(sentence):
                if m.group(1) not in known:
                    report.violations.append(
                        {"type": "unknown_arxiv_id", "sentence": idx, "detail": m.group(1)}
                    )
    if not report.sentences:
        report.violations.append({"type": "empty", "sentence": -1, "detail": "no sentences"})
    return report


def correction_message(report: CitationReport) -> str:
    """Follow-up user turn asking the model to fix its own citation violations."""
    lines = []
    for v in report.violations[:8]:
        lines.append(f"- {v['type']}: {v['detail']}")
    k = report.k
    return (
        "Your answer broke the citation rules:\n"
        + "\n".join(lines)
        + f"\n\nRewrite the answer. Every sentence that states a fact must end with at least one "
        f"marker [n] with n between 1 and {k} (placed before the final period). "
        "Do not write arXiv IDs. Use only the context. If the context cannot answer the "
        f"question, reply with exactly {REFUSAL_SENTINEL}."
    )


# --------------------------------------------------------------------------- #
# Enforcement                                                                 #
# --------------------------------------------------------------------------- #
def _normalise_markers(sentence: str, k: int) -> str:
    """Drop out-of-range numbers, rewrite every marker as ``[a][b]``."""

    def repl(m: "re.Match[str]") -> str:
        valid = [n for n in parse_marker(m.group(1)) if 1 <= n <= k]
        return "".join(f"[{n}]" for n in valid)

    return re.sub(r"[ \t]{2,}", " ", _CITE_RE.sub(repl, sentence)).strip()


def enforce(
    answer: str, source_ids: Sequence[str], policy: str = "drop", min_words: int = 5
) -> Tuple[str, Dict]:
    """Return ``(clean_answer, info)``.

    ``policy="drop"``  remove factual sentences that have no valid citation;
    ``policy="flag"``  keep them but append `` [uncited]``.
    Invalid markers are always removed.  Paragraph/bullet structure is preserved.
    """
    if policy not in ("drop", "flag"):
        raise ValueError("policy must be 'drop' or 'flag'")
    k = len(source_ids)
    dropped: List[str] = []
    flagged: List[str] = []
    out_chunks: List[str] = []
    for chunk in re.split(r"(\n+)", answer):
        if chunk.strip() == "":
            out_chunks.append(chunk)
            continue
        prefix_m = _BULLET_RE.match(chunk)
        prefix = prefix_m.group(1) if prefix_m else ""
        kept: List[str] = []
        for sentence in _split_line(chunk[len(prefix):].strip()):
            numbers = extract_numbers(sentence)
            has_valid = any(1 <= n <= k for n in numbers)
            if needs_citation(sentence, min_words) and not has_valid:
                if policy == "drop":
                    dropped.append(sentence)
                    continue
                flagged.append(sentence)
                kept.append(_normalise_markers(sentence, k) + " [uncited]")
                continue
            kept.append(_normalise_markers(sentence, k))
        out_chunks.append(prefix + " ".join(kept) if kept else "")
    text = "".join(out_chunks)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    info = {
        "policy": policy,
        "dropped_sentences": dropped,
        "flagged_sentences": flagged,
        "empty_after_enforcement": not text,
    }
    return text, info
