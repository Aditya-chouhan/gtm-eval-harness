#!/usr/bin/env python3
"""Check whether the numbers in a repo's README appear in its committed data.

A write-up that cites figures a reader cannot find in the repository is asking
to be taken on trust. This surfaces every numeric claim in a README that does
not appear in any committed data file, so each one can be checked by a human.

**This is a triage tool, not a verdict.** A README legitimately contains
numbers that will never appear in a data file -- version pins, retry counts,
install sizes, dates, ordinal list items. A flagged claim means "go look at
this one", not "this is false". The useful signal is a claim that *should*
have been backed by the committed output and wasn't.

Sources are restricted to committed data files (.json/.csv/.tsv/.txt) found via
`git ls-files`, because the question being asked is specifically whether a
reader of the *published* repository can verify the number.

Usage:
    python scripts/check_readme_grounding.py <repo> [<repo> ...]
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Tuple

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from gtm_eval.grounding import check_numeric_grounding, grounding_rate
from gtm_eval.metrics import wilson_interval

DATA_SUFFIXES = (".json", ".csv", ".tsv", ".txt")
MAX_SOURCE_BYTES = 5_000_000


def tracked_files(repo: str) -> List[str]:
    try:
        out = subprocess.run(
            ["git", "-C", repo, "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [line for line in out.splitlines() if line.strip()]


def load_sources(repo: str, files: List[str]) -> Tuple[List[str], List[str]]:
    texts, names = [], []
    for rel in files:
        if not rel.lower().endswith(DATA_SUFFIXES):
            continue
        path = os.path.join(repo, rel)
        try:
            if os.path.getsize(path) > MAX_SOURCE_BYTES:
                continue
            with open(path, encoding="utf-8", errors="replace") as handle:
                texts.append(handle.read())
            names.append(rel)
        except OSError:
            continue
    return texts, names


def check_repo(repo: str) -> None:
    name = os.path.basename(os.path.abspath(repo.rstrip("/")))
    readme_path = os.path.join(repo, "README.md")

    print("=" * 68)
    print(name)
    print("=" * 68)

    if not os.path.exists(readme_path):
        print("  no README.md -- skipped\n")
        return

    files = tracked_files(repo)
    if not files:
        print("  not a git repo, or nothing tracked -- skipped\n")
        return

    with open(readme_path, encoding="utf-8", errors="replace") as handle:
        readme = handle.read()

    sources, source_names = load_sources(repo, files)
    if not sources:
        print("  no committed data files (.json/.csv/.tsv/.txt) to verify against.")
        print("  Every numeric claim in this README is unverifiable from the repo.\n")
        return

    verdicts = check_numeric_grounding(readme, sources)
    supported, total = grounding_rate(verdicts)

    if total == 0:
        print("  README makes no numeric claims.\n")
        return

    interval = wilson_interval(supported, total)
    print(
        "  {}/{} numeric claims appear in committed data ({:.0%})  {}".format(
            supported, total, supported / total, interval
        )
    )
    print("  sources: {} file(s) -- {}".format(
        len(source_names), ", ".join(source_names[:4])
        + (", ..." if len(source_names) > 4 else "")
    ))

    unsupported = [v for v in verdicts if not v.supported]
    if not unsupported:
        print("\n  Every numeric claim traces to committed data.\n")
        return

    print("\n  NOT FOUND IN COMMITTED DATA ({}) -- each needs a human look:".format(
        len(unsupported)
    ))
    for verdict in unsupported:
        print("    {:<12} {}".format(
            repr(verdict.claim.raw), verdict.claim.context(readme, 34)
        ))
    print()


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    for repo in argv[1:]:
        check_repo(repo)
    print("Reminder: a flagged claim means 'check this', not 'this is false'.")
    print("Version pins, retry counts and install sizes will always flag.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
