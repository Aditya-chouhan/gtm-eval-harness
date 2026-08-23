#!/usr/bin/env python3
"""Re-analyse the published groundtruth verification run.

groundtruth reported 26 true positives out of 43 hand-checked candidates and
published the resulting 60.5% precision as-is, false positives included. That
is already more honest than most published numbers.

This script asks the next two questions of it:

  1. How wide is the interval around 60.5% at n=43?
  2. How much of the sample came from one repository?

Reads `output/verified_findings.json` from a groundtruth checkout. Makes no
network calls and needs no API key.

Usage:
    python scripts/analyze_groundtruth.py [path/to/groundtruth]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from gtm_eval.metrics import ConfusionMatrix, accuracy, precision, wilson_interval
from gtm_eval.report import format_report, stratify

DEFAULT_REPO = os.path.expanduser("~/Projects/groundtruth")
RELATIVE_PATH = os.path.join("output", "verified_findings.json")


def load_verdicts(repo_root: str):
    path = os.path.join(repo_root, RELATIVE_PATH)
    if not os.path.exists(path):
        raise SystemExit(
            "Could not find {}\n"
            "Pass the path to a groundtruth checkout as the first argument.".format(
                path
            )
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle), path


def main(argv):
    repo_root = argv[1] if len(argv) > 1 else DEFAULT_REPO
    data, path = load_verdicts(repo_root)
    verdicts = data.get("verdicts", [])
    if not verdicts:
        raise SystemExit("No verdicts found in {}".format(path))

    tp = sum(1 for v in verdicts if v.get("verdict") == "true_positive")
    fp = sum(1 for v in verdicts if v.get("verdict") == "false_positive")
    cm = ConfusionMatrix(tp=tp, fp=fp)

    print("Source: {}".format(path))
    print("Scanned at: {}".format(data.get("scanned_at", "unknown")))
    print("Repos scanned: {}".format(data.get("repos_scanned", "unknown")))
    print()

    point = precision(cm)
    interval = wilson_interval(tp, tp + fp)
    print("PRECISION")
    print("  point estimate : {:.1%}  ({}/{})".format(point, tp, tp + fp))
    print("  95% interval   : {}".format(interval))
    print("  interval width : {:.1f} percentage points".format(
        (interval.high - interval.low) * 100
    ))
    print()
    print("  Reading: the published 60.5% is the centre of a range that")
    print("  plausibly runs from about {:.0%} to {:.0%}. At n={} the sample".format(
        interval.low, interval.high, tp + fp
    ))
    print("  cannot distinguish 'about half' from 'about three quarters'.")
    print()

    print("RECALL AND ACCURACY")
    print("  recall   : not computable -- the scan only surfaces candidates,")
    print("             so the misses it never flagged were never labelled.")
    print("  accuracy : {}".format(
        "not computable (no true negatives observed)"
        if accuracy(cm) is None
        else "{:.1%}".format(accuracy(cm))
    ))
    print()

    pairs = [
        (v.get("repo", "unknown"), v.get("verdict") == "true_positive")
        for v in verdicts
    ]
    print(format_report("PRECISION BY REPOSITORY", stratify(pairs)))
    print()
    print("Every number above recomputes from the committed JSON with no")
    print("network access. Re-run this script to reproduce it.")


if __name__ == "__main__":
    main(sys.argv)
