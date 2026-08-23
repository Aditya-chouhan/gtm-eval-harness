"""Stratified reporting over a labelled evaluation set.

A single pooled precision number hides two things that change how much you
should trust it:

* **Sample size.** 60.5% from 43 items and 60.5% from 4,300 items are different
  claims. Every rate here ships with a Wilson interval.
* **Concentration.** If most of the sample comes from one source, the pooled
  number largely describes that source. `stratify` breaks the result down by
  group and `concentration` reports how lopsided the sample is, so a result
  that is really "one repo's number" cannot be quietly presented as a general
  one.

Standard library only. Python 3.9 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics import Interval, wilson_interval

__all__ = [
    "GroupResult",
    "stratify",
    "concentration",
    "format_rate",
    "format_report",
]


@dataclass(frozen=True)
class GroupResult:
    """Success rate for one slice of the evaluation set."""

    name: str
    successes: int
    trials: int
    interval: Optional[Interval]

    @property
    def rate(self) -> Optional[float]:
        if self.trials == 0:
            return None
        return self.successes / self.trials

    @property
    def share_of(self) -> int:
        return self.trials


def stratify(
    items: Iterable[Tuple[str, bool]], confidence: float = 0.95
) -> List[GroupResult]:
    """Group `(group_name, was_correct)` pairs and score each group.

    Returned largest-group-first, which is the order that makes concentration
    visible at a glance.
    """
    successes: Dict[str, int] = {}
    trials: Dict[str, int] = {}
    for name, correct in items:
        trials[name] = trials.get(name, 0) + 1
        successes[name] = successes.get(name, 0) + (1 if correct else 0)

    results = [
        GroupResult(
            name=name,
            successes=successes[name],
            trials=trials[name],
            interval=wilson_interval(successes[name], trials[name], confidence),
        )
        for name in trials
    ]
    results.sort(key=lambda r: (-r.trials, r.name))
    return results


def concentration(results: Sequence[GroupResult]) -> Optional[float]:
    """Share of the sample contributed by the single largest group.

    1.0 means everything came from one source. Above ~0.5, the pooled rate is
    mostly a statement about that one group.
    """
    total = sum(r.trials for r in results)
    if total == 0:
        return None
    return max(r.trials for r in results) / total


def format_rate(
    successes: int, trials: int, confidence: float = 0.95
) -> str:
    """Render a rate as point estimate plus interval, never bare."""
    if trials == 0:
        return "n/a (no observations)"
    rate = successes / trials
    interval = wilson_interval(successes, trials, confidence)
    return "{:.1%} ({}/{})  {}".format(rate, successes, trials, interval)


def format_report(
    title: str,
    results: Sequence[GroupResult],
    confidence: float = 0.95,
    concentration_threshold: float = 0.5,
) -> str:
    """Human-readable stratified report, with an explicit caveat when the
    sample is dominated by one group."""
    total_trials = sum(r.trials for r in results)
    total_successes = sum(r.successes for r in results)

    lines: List[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    lines.append("POOLED")
    lines.append("  " + format_rate(total_successes, total_trials, confidence))
    lines.append("")
    lines.append("BY GROUP (largest first)")

    width = max((len(r.name) for r in results), default=0)
    for r in results:
        rate_text = "n/a" if r.rate is None else "{:>6.1%}".format(r.rate)
        interval_text = "" if r.interval is None else "  {}".format(r.interval)
        lines.append(
            "  {name:<{w}}  {rate}  ({s}/{t}){iv}".format(
                name=r.name,
                w=width,
                rate=rate_text,
                s=r.successes,
                t=r.trials,
                iv=interval_text,
            )
        )

    conc = concentration(results)
    if conc is not None:
        lines.append("")
        lines.append("SAMPLE SHAPE")
        largest = results[0]
        lines.append(
            "  {:.0%} of the sample is a single group ({}).".format(conc, largest.name)
        )
        # Strictly greater: an even two-way split lands exactly on 0.5 and is
        # the best case, not a warning condition. The claim being made is
        # "more than half the sample is one group".
        if conc > concentration_threshold:
            lines.append(
                "  WARNING: the pooled figure above is substantially a statement"
            )
            lines.append(
                "  about {} and should not be presented as a general rate.".format(
                    largest.name
                )
            )
    return "\n".join(lines)
