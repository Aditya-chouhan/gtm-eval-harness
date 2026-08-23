"""Tests for gtm_eval.grounding.

Two things are being pinned down here: that formatting differences between a
generated claim and its source do not cause false alarms, and that genuinely
different claims are never allowed to satisfy each other.
"""

from __future__ import annotations

import pytest

from gtm_eval.grounding import (
    GroundingStatus,
    LLMClaimChecker,
    NumberKind,
    check_numeric_grounding,
    extract_numbers,
    format_grounding_report,
    grounding_rate,
)


def kinds_and_values(text):
    return [(n.kind, n.value) for n in extract_numbers(text)]


class TestExtraction:
    def test_plain_count(self):
        assert kinds_and_values("We have 12 reps") == [(NumberKind.COUNT, 12.0)]

    def test_currency_symbol(self):
        assert kinds_and_values("raised $50") == [(NumberKind.CURRENCY, 50.0)]

    def test_percent_symbol_and_word(self):
        assert kinds_and_values("grew 45%") == [(NumberKind.PERCENT, 45.0)]
        assert kinds_and_values("grew 45 percent") == [(NumberKind.PERCENT, 45.0)]

    def test_year_detected(self):
        assert kinds_and_values("founded in 2019") == [(NumberKind.YEAR, 2019.0)]

    def test_four_digit_outside_year_range_is_count(self):
        assert kinds_and_values("shipped 5200 units") == [(NumberKind.COUNT, 5200.0)]

    def test_comma_grouped_number_is_not_a_year(self):
        # "2,024" is a quantity, not a year.
        assert kinds_and_values("2,024 signups") == [(NumberKind.COUNT, 2024.0)]

    def test_magnitude_suffixes(self):
        assert kinds_and_values("$50M")[0][1] == pytest.approx(5e7)
        assert kinds_and_values("$1.2B")[0][1] == pytest.approx(1.2e9)
        assert kinds_and_values("30k users")[0][1] == pytest.approx(3e4)

    def test_magnitude_words(self):
        assert kinds_and_values("$50 million")[0][1] == pytest.approx(5e7)
        assert kinds_and_values("2 billion records")[0][1] == pytest.approx(2e9)

    def test_trailing_currency_word_marks_money(self):
        result = kinds_and_values("50 million dollars")
        assert result == [(NumberKind.CURRENCY, 5e7)]

    def test_rupees_and_other_symbols(self):
        assert kinds_and_values("₹20L")[0][0] is NumberKind.CURRENCY
        assert kinds_and_values("2000 rupees") == [(NumberKind.CURRENCY, 2000.0)]

    def test_multiple_claims_in_one_sentence(self):
        found = kinds_and_values("raised $50M in 2024, up 45% on 1,200 customers")
        assert (NumberKind.CURRENCY, 5e7) in found
        assert (NumberKind.YEAR, 2024.0) in found
        assert (NumberKind.PERCENT, 45.0) in found
        assert (NumberKind.COUNT, 1200.0) in found

    def test_empty_and_number_free_text(self):
        assert extract_numbers("") == []
        assert extract_numbers("no figures at all here") == []

    def test_words_starting_with_magnitude_letters_are_not_magnitudes(self):
        # The regex must not read "members"/"minutes"/"billing" as magnitudes.
        assert kinds_and_values("50 members") == [(NumberKind.COUNT, 50.0)]
        assert kinds_and_values("5 minutes") == [(NumberKind.COUNT, 5.0)]
        assert kinds_and_values("10 b2b leads") == [(NumberKind.COUNT, 10.0)]

    def test_raw_text_and_span_recorded(self):
        value = extract_numbers("we raised $50M last year")[0]
        assert "50" in value.raw
        assert value.start < value.end

    def test_context_window_includes_surrounding_words(self):
        text = "The company raised $50M in a Series B round last spring."
        value = extract_numbers(text)[0]
        assert "raised" in value.context(text)


class TestNormalisationEquivalence:
    @pytest.mark.parametrize(
        "written",
        ["$50M", "$50 million", "$50,000,000", "50 million dollars"],
    )
    def test_all_spellings_of_fifty_million_are_one_identity(self, written):
        target = extract_numbers("$50M")[0].identity
        assert extract_numbers(written)[0].identity == target

    def test_kind_is_part_of_identity(self):
        # $50 and 50% must never satisfy each other.
        money = extract_numbers("$50")[0]
        percent = extract_numbers("50%")[0]
        assert money.value == percent.value
        assert money.identity != percent.identity


class TestGroundingCheck:
    def test_supported_when_value_present(self):
        verdicts = check_numeric_grounding(
            "They raised $50M.", ["Funding: $50,000,000 Series B."]
        )
        assert len(verdicts) == 1
        assert verdicts[0].status is GroundingStatus.SUPPORTED
        assert verdicts[0].evidence is not None

    def test_unsupported_when_value_absent(self):
        verdicts = check_numeric_grounding(
            "They raised $80M.", ["Funding: $50,000,000 Series B."]
        )
        assert verdicts[0].status is GroundingStatus.UNSUPPORTED
        assert verdicts[0].evidence is None

    def test_wrong_kind_is_not_support(self):
        # Source mentions 45 customers; the brief claims 45% growth.
        verdicts = check_numeric_grounding("Grew 45%.", ["They have 45 customers."])
        assert verdicts[0].status is GroundingStatus.UNSUPPORTED

    def test_searches_across_multiple_sources(self):
        verdicts = check_numeric_grounding(
            "Raised $50M in 2019.",
            ["Series B was $50M.", "Founded 2019 in Berlin."],
        )
        assert all(v.supported for v in verdicts)

    def test_repeated_claim_reported_each_time(self):
        verdicts = check_numeric_grounding("$80M. Again, $80M.", ["nothing here"])
        assert len(verdicts) == 2
        assert not any(v.supported for v in verdicts)

    def test_no_claims_yields_no_verdicts(self):
        assert check_numeric_grounding("No numbers here.", ["$50M"]) == []

    def test_empty_sources_make_everything_unsupported(self):
        verdicts = check_numeric_grounding("Raised $50M.", [])
        assert verdicts and not verdicts[0].supported

    def test_documented_limitation_swapped_numbers_pass(self):
        """Known scope limit, asserted so it stays known.

        Both figures are real and both appear in the source, but the brief has
        attached them to the wrong things. This check passes it, and the report
        says so in its SCOPE section.
        """
        # Both figures are real and both appear in the source, but the brief
        # has attached them to the wrong quarters. Every claim passes.
        verdicts = check_numeric_grounding(
            "ARR grew from $50M in Q1 to $200M in Q4.",
            ["Q1 was $200M. Q4 was $50M."],
        )
        assert verdicts
        assert all(v.supported for v in verdicts)


class TestGroundingRate:
    def test_counts_not_bare_float(self):
        verdicts = check_numeric_grounding("$50M and $80M", ["$50M only"])
        assert grounding_rate(verdicts) == (1, 2)

    def test_empty(self):
        assert grounding_rate([]) == (0, 0)


class TestReport:
    def test_reports_rate_with_interval(self):
        verdicts = check_numeric_grounding("$50M and $80M", ["$50M"])
        text = format_grounding_report(verdicts, "$50M and $80M")
        assert "1/2" in text
        assert "CI" in text

    def test_lists_unsupported_claims_with_context(self):
        generated = "The company raised $80M last year."
        text = format_grounding_report(
            check_numeric_grounding(generated, ["raised $50M"]), generated
        )
        assert "UNSUPPORTED CLAIMS" in text
        assert "raised" in text

    def test_clean_result_says_so(self):
        generated = "Raised $50M."
        text = format_grounding_report(
            check_numeric_grounding(generated, ["$50M"]), generated
        )
        assert "None. Every numeric claim traces to a source." in text

    def test_no_claims_message(self):
        text = format_grounding_report([], "no numbers")
        assert "No numeric claims found" in text

    def test_scope_caveat_always_present(self):
        generated = "Raised $50M."
        text = format_grounding_report(
            check_numeric_grounding(generated, ["$50M"]), generated
        )
        assert "SCOPE" in text
        assert "swapped-but-real" in text

    def test_breaks_down_by_kind(self):
        generated = "Raised $50M in 2024, up 45%."
        text = format_grounding_report(
            check_numeric_grounding(generated, ["$50M in 2024"]), generated
        )
        assert "BY CLAIM KIND" in text
        assert "currency" in text and "percent" in text


class TestLLMCheckerSlot:
    def test_raises_rather_than_pretending(self):
        with pytest.raises(NotImplementedError, match="cohens_kappa"):
            LLMClaimChecker().check("they expanded into Europe", ["some source"])
