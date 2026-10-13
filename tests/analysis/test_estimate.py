import numpy as np
import pytest

from madcal.analysis import Estimate, EstimateNull, bootstrap_by_question


def width(estimate):
    assert estimate.ci_low is not None and estimate.ci_high is not None
    return estimate.ci_high - estimate.ci_low


def bounds(estimate):
    assert estimate.ci_low is not None and estimate.ci_high is not None
    return estimate.ci_low, estimate.ci_high


def mean_of(values):
    array = np.asarray(values, dtype=np.float64)

    def statistic(indices):
        if indices.size == 0:
            raise ValueError("no rows")
        return float(np.mean(array[indices]))

    return statistic


def test_point_estimate_is_the_statistic_over_every_row():
    estimate = bootstrap_by_question(
        mean_of([0.0, 1.0, 1.0, 1.0]), ["q1", "q2", "q3", "q4"], n_resamples=0
    )
    assert estimate.value == 0.75
    assert estimate.ci_low is None and estimate.ci_high is None


def test_an_empty_cell_is_a_null_with_a_reason():
    estimate = bootstrap_by_question(mean_of([]), [], n_resamples=10)
    assert estimate.value is None
    assert estimate.reason is EstimateNull.NO_ROWS


def test_a_statistic_undefined_on_the_cell_is_a_null_with_a_reason():
    def always_undefined(indices):
        raise ValueError("one class")

    estimate = bootstrap_by_question(always_undefined, ["q1", "q2"], n_resamples=10)
    assert estimate.value is None
    assert estimate.reason is EstimateNull.UNDEFINED_ON_CELL
    assert estimate.n_resamples_failed == 0


def test_resampling_draws_whole_questions_so_rows_of_one_question_move_together():
    values = [0.0, 0.0, 1.0, 1.0]
    question_ids = ["q1", "q1", "q2", "q2"]
    estimate = bootstrap_by_question(
        mean_of(values), question_ids, n_resamples=500, seed=7, confidence_level=0.99
    )
    assert estimate.value == 0.5
    assert estimate.ci_low == 0.0
    assert estimate.ci_high == 1.0


def test_resampling_by_row_would_have_given_a_narrower_interval():
    values = [float(index % 2) for index in range(30) for _ in range(2)]
    paired = [f"q{index}" for index in range(30) for _ in range(2)]
    unpaired = [f"q{index}" for index in range(60)]
    by_question = bootstrap_by_question(mean_of(values), paired, n_resamples=1000, seed=3)
    by_row = bootstrap_by_question(mean_of(values), unpaired, n_resamples=1000, seed=3)
    assert width(by_question) > width(by_row)


def test_a_constant_statistic_has_a_degenerate_interval():
    estimate = bootstrap_by_question(
        mean_of([1.0, 1.0, 1.0]), ["q1", "q2", "q3"], n_resamples=200, seed=1
    )
    assert estimate.value == 1.0
    assert estimate.ci_low == 1.0
    assert estimate.ci_high == 1.0


def test_resamples_that_fail_are_counted_rather_than_dropped_silently():
    calls = {"n": 0}

    def fails_after_the_point_estimate(indices):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.5
        raise ValueError("one class")

    estimate = bootstrap_by_question(fails_after_the_point_estimate, ["q1", "q2"], n_resamples=25)
    assert estimate.value == 0.5
    assert estimate.n_resamples_failed == 25
    assert estimate.ci_low is None


def test_the_same_seed_gives_the_same_interval():
    first = bootstrap_by_question(mean_of([0.0, 1.0, 1.0]), ["a", "b", "c"], seed=11)
    second = bootstrap_by_question(mean_of([0.0, 1.0, 1.0]), ["a", "b", "c"], seed=11)
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)


def test_a_different_seed_gives_a_different_interval():
    values = [index / 29.0 for index in range(30)]
    question_ids = [f"q{index}" for index in range(30)]
    first = bootstrap_by_question(mean_of(values), question_ids, n_resamples=200, seed=1)
    second = bootstrap_by_question(mean_of(values), question_ids, n_resamples=200, seed=2)
    assert (first.ci_low, first.ci_high) != (second.ci_low, second.ci_high)


def test_a_wider_confidence_level_gives_a_wider_interval():
    narrow = bootstrap_by_question(
        mean_of([0.0, 0.25, 0.5, 0.75, 1.0]),
        ["a", "b", "c", "d", "e"],
        n_resamples=1000,
        seed=5,
        confidence_level=0.5,
    )
    wide = bootstrap_by_question(
        mean_of([0.0, 0.25, 0.5, 0.75, 1.0]),
        ["a", "b", "c", "d", "e"],
        n_resamples=1000,
        seed=5,
        confidence_level=0.99,
    )
    wide_low, wide_high = bounds(wide)
    narrow_low, narrow_high = bounds(narrow)
    assert wide_low <= narrow_low
    assert wide_high >= narrow_high


def test_an_estimate_needs_exactly_one_of_value_or_reason():
    with pytest.raises(ValueError, match="exactly one"):
        Estimate(0.5, None, None, 0, EstimateNull.NO_ROWS)
    with pytest.raises(ValueError, match="exactly one"):
        Estimate(None, None, None, 0, None)


def test_a_null_estimate_carries_no_interval():
    with pytest.raises(ValueError, match="no interval"):
        Estimate(None, 0.1, 0.2, 0, EstimateNull.NO_ROWS)


def test_an_interval_needs_both_bounds():
    with pytest.raises(ValueError, match="both bounds"):
        Estimate(0.5, 0.1, None, 0, None)


def test_an_inverted_interval_is_rejected():
    with pytest.raises(ValueError, match="inverted"):
        Estimate(0.5, 0.9, 0.1, 0, None)


def test_a_negative_resample_count_is_rejected():
    with pytest.raises(ValueError, match="n_resamples must be"):
        bootstrap_by_question(mean_of([1.0]), ["a"], n_resamples=-1)


def test_a_confidence_level_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_by_question(mean_of([1.0]), ["a"], confidence_level=1.0)
