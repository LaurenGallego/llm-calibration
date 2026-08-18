"""SignalValue's invariants.

The type exists so that "no signal could be computed" cannot be mistaken for "the
signal was zero". Every test here is a case where the mistake would otherwise be
silent and would land in a stored row.
"""

import pytest

from caldrift.signals import SignalNull, SignalValue


def test_a_value_and_a_reason_together_are_rejected():
    with pytest.raises(ValueError):
        SignalValue(0.5, SignalNull.NO_CHOICE_SCORES)


def test_neither_a_value_nor_a_reason_is_rejected():
    with pytest.raises(ValueError):
        SignalValue(None)


def test_non_finite_values_are_rejected():
    for bad_value in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError):
            SignalValue(bad_value)
