"""Claim-level grounding checks for generated GTM copy.

The failure this targets is specific: a model writes a brief or an outreach
email and invents a number. "Raised $50M in March" reads exactly as fluently
when it is true and when it is not, and it is the fabrication class that
actually damages a sender's credibility.

Numbers are worth checking mechanically because they are the claims most often
fabricated *and* the claims most cheaply verified. No model is required to ask
whether `$50M` appears anywhere in the source material.

Scope, stated plainly:

* This checks **numeric** claims -- currency, percentages, years, counts. It
  does not check whether a sentence is a fair characterisation of a source, and
  it never will; that judgement needs a reader or a judge model. See
  `LLMClaimChecker` for the interface that slot expects.
* A "supported" verdict means the value appears in the source, not that the
  generated sentence uses it correctly. A brief that swaps two real numbers
  around passes this check and is still wrong.

Both limits are measured rather than asserted -- see `README.md`.

Standard library only. Python 3.9 compatible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .metrics import wilson_interval

__all__ = [
    "NumberKind",
    "NumericValue",
    "GroundingStatus",
    "Verdict",
    "extract_numbers",
    "check_numeric_grounding",
    "grounding_rate",
    "format_grounding_report",
    "LLMClaimChecker",
]


class NumberKind(Enum):
    """What a number is claiming to be.

    Kind is part of identity: `50%` and `$50` are different claims and must not
    satisfy each other.
    """

    CURRENCY = "currency"
    PERCENT = "percent"
    YEAR = "year"
    COUNT = "count"


class GroundingStatus(Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


# Magnitude words and suffixes, normalised to multipliers.
_MAGNITUDES: Dict[str, float] = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mm": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
    # Indian numbering, which appears in this author's own materials
    # ("₹20L lifetime revenue") and in any INR-denominated source.
    "l": 1e5,
    "lac": 1e5,
    "lakh": 1e5,
    "lakhs": 1e5,
    "cr": 1e7,
    "crore": 1e7,
    "crores": 1e7,
}

_CURRENCY_SYMBOLS = "$€£₹¥"

_NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z0-9])                # not a digit embedded in an identifier;
                                    # without this, "b2b" yields "2b" = 2bn
    (?P<currency>[{sym}])?          # optional leading currency symbol
    \s*
    (?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)
    \s*
                                    # longest-first so "million" is not shadowed
                                    # by "m"; relying on backtracking here is
                                    # correct but fragile to later edits
    (?P<mag>thousand|million|billion|trillion|lakhs|lakh|crores|crore|lac
           |mm|bn|cr|k|m|b|t|l)?\b
    \s*
    (?P<pct>%|percent\b)?
    """.format(sym=_CURRENCY_SYMBOLS),
    re.VERBOSE | re.IGNORECASE,
)

# Trailing words that mark an amount as money even without a symbol.
_CURRENCY_WORDS_RE = re.compile(
    r"\s*(?:dollars?|usd|eur|euros?|gbp|pounds?|inr|rupees?)\b", re.IGNORECASE
)

# A bare 4-digit number in this range is treated as a year rather than a count.
_YEAR_MIN, _YEAR_MAX = 1900, 2100


@dataclass(frozen=True)
class NumericValue:
    """One numeric claim, normalised so formatting differences don't matter.

    `$50M`, `50 million dollars` and `$50,000,000` all normalise to the same
    (CURRENCY, 5e7) identity, so a check does not fail merely because the
    generator and the source wrote the figure differently.
    """

    raw: str
    value: float
    kind: NumberKind
    start: int
    end: int

    @property
    def identity(self) -> Tuple[str, float]:
        """What must match for two claims to be the same claim."""
        return (self.kind.value, round(self.value, 6))

    def context(self, text: str, window: int = 45) -> str:
        """Surrounding text, for human review of a flagged claim."""
        left = max(0, self.start - window)
        right = min(len(text), self.end + window)
        snippet = text[left:right].replace("\n", " ").strip()
        prefix = "..." if left > 0 else ""
        suffix = "..." if right < len(text) else ""
        return "{}{}{}".format(prefix, snippet, suffix)


def _classify(
    raw_number: str, currency: Optional[str], magnitude: Optional[str],
    percent: Optional[str], trailing_currency: bool,
) -> Tuple[float, NumberKind]:
    value = float(raw_number.replace(",", ""))
    had_magnitude = False
    if magnitude:
        value *= _MAGNITUDES[magnitude.lower()]
        had_magnitude = True

    if percent:
        return value, NumberKind.PERCENT
    if currency or trailing_currency:
        return value, NumberKind.CURRENCY

    is_bare_integer = "," not in raw_number and "." not in raw_number
    if (
        not had_magnitude
        and is_bare_integer
        and len(raw_number) == 4
        and _YEAR_MIN <= value <= _YEAR_MAX
    ):
        return value, NumberKind.YEAR

    return value, NumberKind.COUNT


def extract_numbers(text: str) -> List[NumericValue]:
    """Pull every numeric claim out of `text`, normalised.

    Overlapping matches are impossible by construction: the scanner advances
    past each match, so `$1.2M` yields one CURRENCY claim rather than a
    currency claim and a stray `2`.
    """
    if not text:
        return []

    found: List[NumericValue] = []
    for match in _NUMBER_RE.finditer(text):
        number = match.group("num")
        if number is None:
            continue

        end = match.end()
        trailing = _CURRENCY_WORDS_RE.match(text, end)
        if trailing and not match.group("pct"):
            end = trailing.end()

        value, kind = _classify(
            raw_number=number,
            currency=match.group("currency"),
            magnitude=match.group("mag"),
            percent=match.group("pct"),
            trailing_currency=bool(trailing) and not match.group("pct"),
        )
        found.append(
            NumericValue(
                raw=text[match.start():end].strip(),
                value=value,
                kind=kind,
                start=match.start(),
                end=end,
            )
        )
    return found


@dataclass(frozen=True)
class Verdict:
    """The outcome of checking one claim against source material."""

    claim: NumericValue
    status: GroundingStatus
    evidence: Optional[str] = None

    @property
    def supported(self) -> bool:
        return self.status is GroundingStatus.SUPPORTED


def check_numeric_grounding(
    generated: str, sources: Sequence[str]
) -> List[Verdict]:
    """Check every numeric claim in `generated` against `sources`.

    A claim is SUPPORTED when a value of the same kind and magnitude appears in
    at least one source document. Duplicate claims are each reported, because
    a number repeated three times in an email is three chances to be wrong in
    front of a reader.
    """
    source_values: Dict[Tuple[str, float], str] = {}
    for source in sources:
        for value in extract_numbers(source):
            source_values.setdefault(value.identity, value.context(source))

    verdicts: List[Verdict] = []
    for claim in extract_numbers(generated):
        evidence = source_values.get(claim.identity)
        verdicts.append(
            Verdict(
                claim=claim,
                status=(
                    GroundingStatus.SUPPORTED
                    if evidence is not None
                    else GroundingStatus.UNSUPPORTED
                ),
                evidence=evidence,
            )
        )
    return verdicts


def grounding_rate(verdicts: Sequence[Verdict]) -> Tuple[int, int]:
    """Return `(supported, total)`.

    Deliberately returns counts rather than a bare float so the caller cannot
    render a rate without the denominator that qualifies it.
    """
    return sum(1 for v in verdicts if v.supported), len(verdicts)


def format_grounding_report(
    verdicts: Sequence[Verdict], generated: str, confidence: float = 0.95
) -> str:
    """Human-readable grounding report, unsupported claims listed in full."""
    supported, total = grounding_rate(verdicts)

    lines: List[str] = ["GROUNDING", "========="]
    if total == 0:
        lines.append("  No numeric claims found in the generated text.")
        lines.append("  This check has nothing to say about non-numeric claims.")
        return "\n".join(lines)

    interval = wilson_interval(supported, total, confidence)
    lines.append(
        "  {}/{} numeric claims supported ({:.1%})  {}".format(
            supported, total, supported / total, interval
        )
    )
    lines.append("")

    by_kind: Dict[NumberKind, List[Verdict]] = {}
    for verdict in verdicts:
        by_kind.setdefault(verdict.claim.kind, []).append(verdict)

    lines.append("BY CLAIM KIND")
    for kind in NumberKind:
        group = by_kind.get(kind)
        if not group:
            continue
        ok = sum(1 for v in group if v.supported)
        lines.append(
            "  {:<9} {}/{}".format(kind.value, ok, len(group))
        )

    unsupported = [v for v in verdicts if not v.supported]
    lines.append("")
    if not unsupported:
        lines.append("UNSUPPORTED CLAIMS")
        lines.append("  None. Every numeric claim traces to a source.")
    else:
        lines.append("UNSUPPORTED CLAIMS ({})".format(len(unsupported)))
        for verdict in unsupported:
            lines.append("  - {!r}".format(verdict.claim.raw))
            lines.append("      {}".format(verdict.claim.context(generated)))

    lines.append("")
    lines.append("SCOPE")
    lines.append("  Numeric claims only. A supported verdict means the value")
    lines.append("  appears in a source, not that the sentence uses it")
    lines.append("  correctly -- swapped-but-real numbers pass this check.")
    return "\n".join(lines)


class LLMClaimChecker:
    """Interface for the judge-model slot. Not implemented.

    Non-numeric claims ("they are expanding into Europe") need a reader or a
    judge model. That slot is defined here so the deterministic checker above
    can be composed with one, and so the shape of the missing piece is explicit
    rather than implied.

    Any implementation must be validated against human labels using
    `gtm_eval.metrics.cohens_kappa` before its output is reported as a rate. An
    unvalidated judge is an opinion with a percent sign attached.
    """

    def check(self, claim: str, sources: Sequence[str]) -> Verdict:
        raise NotImplementedError(
            "No judge model is implemented. Validate any implementation against "
            "human labels with cohens_kappa before reporting its output."
        )
