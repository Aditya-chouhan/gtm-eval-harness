"""Tests for gtm_eval.report.

The behaviour under test is mostly about refusing to let a pooled number stand
unqualified: rates always carry an interval, and a lopsided sample raises a
visible warning.
"""

from __future__ import annotations

import pytest

from gtm_eval.report import (
    GroupResult,
    concentration,
    format_rate,
    format_report,
    stratify,
)


class TestStratify:
    def test_groups_and_counts(self):
        results = stratify(
            [("a", True), ("a", False), ("b", True), ("b", True), ("b", True)]
        )
        by_name = {r.name: r for r in results}
        assert by_name["a"].successes == 1 and by_name["a"].trials == 2
        assert by_name["b"].successes == 3 and by_name["b"].trials == 3

    def test_sorted_largest_group_first(self):
        results = stratify(
            [("small", True), ("big", True), ("big", False), ("big", True)]
        )
        assert [r.name for r in results] == ["big", "small"]

    def test_ties_broken_alphabetically(self):
        results = stratify([("zebra", True), ("alpha", True)])
        assert [r.name for r in results] == ["alpha", "zebra"]

    def test_every_group_gets_an_interval(self):
        for r in stratify([("a", True), ("a", False), ("b", True)]):
            assert r.interval is not None

    def test_rate_computed(self):
        result = stratify([("a", True), ("a", False)])[0]
        assert result.rate == pytest.approx(0.5)

    def test_empty_input(self):
        assert stratify([]) == []


class TestConcentration:
    def test_single_group_is_total(self):
        assert concentration(stratify([("a", True), ("a", False)])) == pytest.approx(1.0)

    def test_even_split(self):
        assert concentration(stratify([("a", True), ("b", True)])) == pytest.approx(0.5)

    def test_matches_groundtruth_shape(self):
        # 28 of 43 candidates came from one repo in the real run.
        items = [("mem0", True)] * 28 + [("other", True)] * 15
        assert concentration(stratify(items)) == pytest.approx(28 / 43, abs=1e-6)

    def test_empty_is_none(self):
        assert concentration([]) is None


class TestFormatRate:
    def test_includes_interval_not_just_point(self):
        text = format_rate(26, 43)
        assert "60.5%" in text
        assert "(26/43)" in text
        assert "95% CI" in text

    def test_zero_trials_is_explicit(self):
        assert "n/a" in format_rate(0, 0)


class TestFormatReport:
    def _lopsided(self):
        return stratify([("dominant", True)] * 9 + [("minor", False)])

    def test_contains_pooled_and_per_group(self):
        text = format_report("TITLE", self._lopsided())
        assert "POOLED" in text
        assert "BY GROUP" in text
        assert "dominant" in text and "minor" in text

    def test_warns_when_one_group_dominates(self):
        text = format_report("TITLE", self._lopsided())
        assert "WARNING" in text
        assert "dominant" in text

    def test_no_warning_on_balanced_sample(self):
        balanced = stratify(
            [("a", True), ("a", False), ("b", True), ("b", False)]
        )
        assert "WARNING" not in format_report("TITLE", balanced)

    def test_threshold_is_configurable(self):
        balanced = stratify([("a", True), ("b", False)])
        strict = format_report("T", balanced, concentration_threshold=0.4)
        assert "WARNING" in strict

    def test_handles_group_with_no_interval_gracefully(self):
        # Defensive: a zero-trial group should not crash rendering.
        results = [GroupResult(name="empty", successes=0, trials=0, interval=None)]
        text = format_report("TITLE", results)
        assert "n/a" in text
