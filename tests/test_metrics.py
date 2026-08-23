"""Tests for gtm_eval.metrics.

The edge cases matter more than the happy path here. Every "undefined returns
None" test below corresponds to a way a metric can silently flatter a system:
precision on zero predictions, accuracy with no observed negatives, kappa on a
unanimous set.
"""

from __future__ import annotations

import math

import pytest

from gtm_eval.metrics import (
    ConfusionMatrix,
    accuracy,
    cohens_kappa,
    confusion_matrix,
    f1,
    precision,
    raw_agreement,
    recall,
    wilson_interval,
)


class TestConfusionMatrix:
    def test_counts_each_quadrant(self):
        cm = confusion_matrix(
            predicted=[True, True, False, False],
            actual=[True, False, True, False],
        )
        assert (cm.tp, cm.fp, cm.fn, cm.tn) == (1, 1, 1, 1)

    def test_derived_totals(self):
        cm = ConfusionMatrix(tp=3, fp=2, fn=4, tn=1)
        assert cm.predicted_positive == 5
        assert cm.actual_positive == 7
        assert cm.total == 10

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            confusion_matrix(predicted=[True], actual=[True, False])

    def test_rejects_negative_counts(self):
        with pytest.raises(ValueError, match="non-negative"):
            ConfusionMatrix(tp=-1)

    def test_rejects_bool_masquerading_as_int(self):
        # bool is a subclass of int; letting it through would make
        # ConfusionMatrix(tp=True) silently mean tp=1.
        with pytest.raises(TypeError):
            ConfusionMatrix(tp=True)

    def test_empty_input_is_all_zeros(self):
        cm = confusion_matrix(predicted=[], actual=[])
        assert cm.total == 0


class TestPrecisionRecallF1:
    def test_known_values(self):
        cm = ConfusionMatrix(tp=6, fp=2, fn=2, tn=10)
        assert precision(cm) == pytest.approx(0.75)
        assert recall(cm) == pytest.approx(0.75)
        assert f1(cm) == pytest.approx(0.75)

    def test_f1_is_harmonic_not_arithmetic_mean(self):
        # precision 1.0, recall 0.5 -> harmonic 0.667, arithmetic would be 0.75
        cm = ConfusionMatrix(tp=5, fp=0, fn=5, tn=0)
        assert precision(cm) == pytest.approx(1.0)
        assert recall(cm) == pytest.approx(0.5)
        assert f1(cm) == pytest.approx(2 / 3)

    def test_precision_undefined_when_nothing_predicted(self):
        # The critical case: a system that flags nothing has undefined
        # precision, not 0.0 and not 1.0.
        cm = ConfusionMatrix(tp=0, fp=0, fn=7, tn=3)
        assert precision(cm) is None

    def test_recall_undefined_when_nothing_to_find(self):
        cm = ConfusionMatrix(tp=0, fp=4, fn=0, tn=6)
        assert recall(cm) is None

    def test_f1_none_if_either_component_undefined(self):
        assert f1(ConfusionMatrix(tp=0, fp=0, fn=3)) is None
        assert f1(ConfusionMatrix(tp=0, fp=3, fn=0)) is None

    def test_f1_none_when_both_are_zero(self):
        cm = ConfusionMatrix(tp=0, fp=5, fn=5)
        assert precision(cm) == 0.0
        assert recall(cm) == 0.0
        assert f1(cm) is None


class TestAccuracy:
    def test_computed_when_negatives_observed(self):
        cm = ConfusionMatrix(tp=4, fp=1, fn=1, tn=4)
        assert accuracy(cm) == pytest.approx(0.8)

    def test_undefined_without_true_negatives(self):
        # A candidate-surfacing scanner never observes what it skipped, so
        # accuracy would describe a population the system never ran against.
        cm = ConfusionMatrix(tp=26, fp=17, fn=0, tn=0)
        assert accuracy(cm) is None


class TestWilsonInterval:
    def test_groundtruth_real_numbers(self):
        """The actual groundtruth result: 26 true positives out of 43 checked.

        The point estimate is 60.5%. The interval is roughly [45.6%, 73.6%] --
        nearly 28 points wide. Reporting 60.5% alone implies a precision the
        sample size does not support.
        """
        interval = wilson_interval(successes=26, trials=43, confidence=0.95)
        assert interval is not None
        assert interval.low == pytest.approx(0.4558, abs=1e-3)
        assert interval.high == pytest.approx(0.7364, abs=1e-3)
        assert interval.high - interval.low > 0.25

    def test_interval_contains_point_estimate(self):
        interval = wilson_interval(26, 43)
        assert interval.low < 26 / 43 < interval.high

    def test_narrows_as_sample_grows(self):
        small = wilson_interval(60, 100)
        large = wilson_interval(600, 1000)
        assert (large.high - large.low) < (small.high - small.low)

    def test_stays_within_zero_one_at_extremes(self):
        perfect = wilson_interval(10, 10)
        assert perfect.high <= 1.0
        assert perfect.low > 0.0

        zero = wilson_interval(0, 10)
        assert zero.low >= 0.0
        assert zero.high < 1.0

    def test_no_trials_is_none(self):
        assert wilson_interval(0, 0) is None

    def test_rejects_impossible_counts(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            wilson_interval(11, 10)

    def test_rejects_unsupported_confidence(self):
        with pytest.raises(ValueError, match="not supported"):
            wilson_interval(5, 10, confidence=0.93)

    def test_higher_confidence_widens(self):
        c90 = wilson_interval(26, 43, confidence=0.90)
        c99 = wilson_interval(26, 43, confidence=0.99)
        assert (c99.high - c99.low) > (c90.high - c90.low)

    def test_str_is_readable(self):
        assert "95% CI" in str(wilson_interval(26, 43))


class TestAgreement:
    def test_perfect_agreement(self):
        labels = [True, False, True, False]
        assert raw_agreement(labels, labels) == pytest.approx(1.0)
        assert cohens_kappa(labels, labels) == pytest.approx(1.0)

    def test_skewed_set_inflates_raw_agreement(self):
        """The reason kappa is reported alongside raw agreement.

        Two raters who both almost always say False agree on 8 of 10 items,
        which looks strong. Chance-corrected, they have essentially nothing.
        """
        human = [False] * 9 + [True]
        judge = [False] * 8 + [True, False]

        assert raw_agreement(human, judge) == pytest.approx(0.8)
        kappa = cohens_kappa(human, judge)
        assert kappa is not None
        assert kappa < 0.3

    def test_worse_than_chance_is_negative(self):
        a = [True, True, False, False]
        b = [False, False, True, True]
        assert cohens_kappa(a, b) < 0

    def test_unanimous_and_identical_is_one(self):
        labels = [True] * 5
        assert cohens_kappa(labels, labels) == pytest.approx(1.0)

    def test_unanimous_but_opposite_is_zero(self):
        assert cohens_kappa([True] * 5, [False] * 5) == pytest.approx(0.0)

    def test_empty_is_none(self):
        assert raw_agreement([], []) is None
        assert cohens_kappa([], []) is None

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            raw_agreement([True], [True, False])


def test_kappa_matches_hand_computed_value():
    """Worked example, verified by hand.

    n=10, raters agree on 7. Rater A says True 5/10, rater B says True 6/10.
    expected = .5*.6 + .5*.4 = .5  ->  kappa = (.7 - .5) / (1 - .5) = 0.4
    """
    a = [True, True, True, True, True, False, False, False, False, False]
    b = [True, True, True, True, False, True, True, False, False, False]

    assert raw_agreement(a, b) == pytest.approx(0.7)
    assert sum(a) == 5 and sum(b) == 6
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_metrics_never_silently_return_zero_for_undefined():
    """Guard against a regression that would reintroduce flattering defaults."""
    empty = ConfusionMatrix()
    assert precision(empty) is None
    assert recall(empty) is None
    assert f1(empty) is None
    assert accuracy(empty) is None
    assert not any(
        value == 0.0
        for value in (precision(empty), recall(empty), f1(empty), accuracy(empty))
        if value is not None
    )


def test_module_has_no_third_party_imports():
    """The core metric layer must stay stdlib-only so it runs anywhere."""
    import gtm_eval.metrics as m

    source = open(m.__file__, encoding="utf-8").read()
    for banned in ("import numpy", "import scipy", "import pandas", "import sklearn"):
        assert banned not in source, "core metrics must remain dependency-free"
    assert math is not None  # module under test uses stdlib math
