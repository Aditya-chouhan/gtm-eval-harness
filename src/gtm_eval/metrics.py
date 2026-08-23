"""Evaluation metrics for GTM AI outputs.

Design decisions worth stating, because they are the point of this module:

1. **Undefined is not zero.** Precision with no positive predictions is
   undefined, not 0.0. Returning 0.0 there silently converts "we never guessed"
   into "we guessed and were always wrong", which is a different claim. These
   functions return None instead.

2. **Point estimates are not reported alone.** At n=43, "60.5% precision" is a
   number with roughly +/-15 points of slack. Any headline proportion here
   ships with a Wilson score interval so the sample size is visible in the
   claim itself rather than buried in a footnote.

3. **Agreement is measured against chance.** When an LLM judge is checked
   against human labels, raw agreement is inflated by the base rate. Cohen's
   kappa is reported alongside it.

Standard library only. Python 3.9 compatible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = [
    "ConfusionMatrix",
    "Interval",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "wilson_interval",
    "raw_agreement",
    "cohens_kappa",
]


@dataclass(frozen=True)
class Interval:
    """A confidence interval on a proportion."""

    low: float
    high: float
    confidence: float

    def __str__(self) -> str:
        pct = int(round(self.confidence * 100))
        return "[{:.1%}, {:.1%}] ({}% CI)".format(self.low, self.high, pct)


@dataclass(frozen=True)
class ConfusionMatrix:
    """Counts of predicted-vs-actual for a binary decision.

    `tn` is optional in practice: a scanner that only surfaces candidates never
    observes true negatives, so it stays 0 and accuracy stays meaningless. That
    is a property of the task, not a bug, and `accuracy()` returns None rather
    than a flattering number when `tn` is absent.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __post_init__(self) -> None:
        for name in ("tp", "fp", "fn", "tn"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("{} must be an int, got {!r}".format(name, value))
            if value < 0:
                raise ValueError("{} must be non-negative, got {}".format(name, value))

    @property
    def predicted_positive(self) -> int:
        return self.tp + self.fp

    @property
    def actual_positive(self) -> int:
        return self.tp + self.fn

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


def confusion_matrix(
    predicted: Sequence[bool], actual: Sequence[bool]
) -> ConfusionMatrix:
    """Build a confusion matrix from two aligned label sequences."""
    if len(predicted) != len(actual):
        raise ValueError(
            "predicted and actual must be the same length "
            "({} vs {})".format(len(predicted), len(actual))
        )
    tp = fp = fn = tn = 0
    for p, a in zip(predicted, actual):
        if p and a:
            tp += 1
        elif p and not a:
            fp += 1
        elif not p and a:
            fn += 1
        else:
            tn += 1
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def precision(cm: ConfusionMatrix) -> Optional[float]:
    """Of what we flagged, how much was real. None if nothing was flagged."""
    if cm.predicted_positive == 0:
        return None
    return cm.tp / cm.predicted_positive


def recall(cm: ConfusionMatrix) -> Optional[float]:
    """Of what was real, how much we caught. None if there was nothing to catch."""
    if cm.actual_positive == 0:
        return None
    return cm.tp / cm.actual_positive


def f1(cm: ConfusionMatrix) -> Optional[float]:
    """Harmonic mean of precision and recall. None if either is undefined."""
    p = precision(cm)
    r = recall(cm)
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


def accuracy(cm: ConfusionMatrix) -> Optional[float]:
    """Overall correctness.

    Returns None when no true negatives were observed. A candidate-surfacing
    pipeline never sees the negatives it silently skipped, so accuracy computed
    over tp/fp/fn alone would describe a different population than the one the
    system actually runs against.
    """
    if cm.tn == 0:
        return None
    if cm.total == 0:
        return None
    return (cm.tp + cm.tn) / cm.total


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> Optional[Interval]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation, which misbehaves badly at small n
    and near 0 or 1 -- exactly the regime these evaluations run in. Returns
    None when there are no trials.
    """
    if trials < 0 or successes < 0:
        raise ValueError("successes and trials must be non-negative")
    if successes > trials:
        raise ValueError(
            "successes ({}) cannot exceed trials ({})".format(successes, trials)
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if trials == 0:
        return None

    z = _z_for_confidence(confidence)
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denom
    )
    return Interval(
        low=max(0.0, center - margin),
        high=min(1.0, center + margin),
        confidence=confidence,
    )


def raw_agreement(a: Sequence[bool], b: Sequence[bool]) -> Optional[float]:
    """Fraction of items two raters labelled identically.

    Reported only alongside `cohens_kappa`. On a skewed set, two raters who
    both mostly say False will agree ~90% of the time while sharing no real
    signal, so this number on its own is close to meaningless.
    """
    if len(a) != len(b):
        raise ValueError(
            "label sequences must be the same length ({} vs {})".format(len(a), len(b))
        )
    if not a:
        return None
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> Optional[float]:
    """Chance-corrected agreement between two raters.

    1.0 is perfect, 0.0 is exactly chance, negative is worse than chance.
    Returns None for an empty set, and 1.0 when both raters were unanimous and
    identical (where the chance-corrected form is otherwise 0/0).
    """
    observed = raw_agreement(a, b)
    if observed is None:
        return None

    n = len(a)
    a_pos = sum(1 for x in a if x) / n
    b_pos = sum(1 for x in b if x) / n
    expected = a_pos * b_pos + (1.0 - a_pos) * (1.0 - b_pos)

    if math.isclose(expected, 1.0):
        # Both raters gave one label to everything. Kappa is undefined here;
        # report perfect agreement only if they actually matched.
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


# Two-sided z scores. Interpolating a normal quantile for arbitrary confidence
# levels would add a dependency for no real benefit; these are the levels
# anyone actually reports.
_Z_SCORES = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.98: 2.3263478740408408,
    0.99: 2.5758293035489004,
}


def _z_for_confidence(confidence: float) -> float:
    key = round(confidence, 4)
    for level, z in _Z_SCORES.items():
        if math.isclose(key, level, rel_tol=0.0, abs_tol=1e-9):
            return z
    supported = ", ".join(str(level) for level in sorted(_Z_SCORES))
    raise ValueError(
        "confidence {} is not supported; use one of: {}".format(confidence, supported)
    )
