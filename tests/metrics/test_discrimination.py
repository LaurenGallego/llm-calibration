"""Hand-computed expected values for AUROC."""

import numpy as np
import pytest

from madcal.metrics import auroc, average_ranks


def test_four_question_case():
    # Correct answers carry confidence 0.35 and 0.80; incorrect carry 0.10 and 0.40.
    # Of the four (correct, incorrect) pairs, three rank the correct one higher:
    # (0.35, 0.10) yes, (0.35, 0.40) no, (0.80, 0.10) yes, (0.80, 0.40) yes -> 3/4.
    assert auroc([0.10, 0.40, 0.35, 0.80], [0, 0, 1, 1]) == pytest.approx(0.75)


def test_a_tied_pair_counts_a_half():
    # Pairs: (0.6, 0.5) yes, (0.6, 0.4) yes, (0.5, 0.5) tie -> 0.5, (0.5, 0.4) yes.
    #                                                    (1 + 1 + 0.5 + 1) / 4 = 0.875
    assert auroc([0.6, 0.5, 0.5, 0.4], [1, 1, 0, 0]) == pytest.approx(0.875)


def test_perfect_separation_is_one():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_perfectly_inverted_separation_is_zero():
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)


def test_all_confidences_tied_is_one_half():
    assert auroc([0.7] * 6, [1, 1, 1, 0, 0, 0]) == pytest.approx(0.5)


def test_it_is_invariant_to_a_monotone_rescaling_of_confidence():
    conf = [0.10, 0.40, 0.35, 0.80]
    correct = [0, 0, 1, 1]
    squared = [value**2 for value in conf]
    assert auroc(squared, correct) == pytest.approx(auroc(conf, correct))


def test_it_is_invariant_to_the_order_of_the_questions():
    conf = [0.10, 0.40, 0.35, 0.80]
    correct = [0, 0, 1, 1]
    order = [3, 0, 2, 1]
    shuffled_conf = [conf[i] for i in order]
    shuffled_correct = [correct[i] for i in order]
    assert auroc(shuffled_conf, shuffled_correct) == pytest.approx(auroc(conf, correct))


def test_a_cell_with_one_class_is_refused_rather_than_reported_as_chance():
    with pytest.raises(ValueError, match="needs both classes"):
        auroc([0.2, 0.5, 0.9], [1, 1, 1])
    with pytest.raises(ValueError, match="needs both classes"):
        auroc([0.2, 0.5, 0.9], [0, 0, 0])


def test_average_ranks_share_a_mean_rank_between_ties():
    ranks = average_ranks(np.asarray([0.4, 0.5, 0.5, 0.6]))
    assert ranks == pytest.approx([1.0, 2.5, 2.5, 4.0])


def test_average_ranks_follow_the_values_not_the_positions():
    ranks = average_ranks(np.asarray([0.6, 0.4, 0.5, 0.5]))
    assert ranks == pytest.approx([4.0, 1.0, 2.5, 2.5])
