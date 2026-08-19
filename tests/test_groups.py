import numpy as np
import pandas as pd

from neu_intellicage.groups import bootstrap_ci, compare, exact_permutation_p


def test_exact_permutation_enumerates_all_splits():
    p, total, floor = exact_permutation_p(np.array([1.0, 2, 3, 4]), np.array([5.0, 6, 7, 8]))
    assert total == 70
    assert np.isclose(floor, 2 / 70)
    assert np.isclose(p, 2 / 70)


def test_four_versus_four_cannot_reach_p_below_min_attainable():
    """The design floor must be surfaced, not hidden behind a small p."""
    a, b = np.array([0.0, 0, 0, 0]), np.array([100.0, 100, 100, 100])
    p, _, floor = exact_permutation_p(a, b)
    assert p >= floor


def test_identical_groups_give_p_of_one():
    p, _, _ = exact_permutation_p(np.array([1.0, 1, 1, 1]), np.array([1.0, 1, 1, 1]))
    assert p == 1.0


def test_bootstrap_ci_is_deterministic_and_brackets_the_difference():
    a, b = np.array([0.5, 0.55, 0.6, 0.52]), np.array([0.45, 0.47, 0.5, 0.44])
    low, high = bootstrap_ci(a, b)
    assert (low, high) == bootstrap_ci(a, b)
    assert low <= a.mean() - b.mean() <= high


def test_compare_reports_effect_size_and_ci_regardless_of_significance():
    values = pd.DataFrame({"AnimalName": [f"Animal {i}" for i in range(1, 9)],
                           "accuracy": [0.5, 0.51, 0.49, 0.5, 0.5, 0.49, 0.51, 0.5]})
    result = compare(values, "accuracy",
                     {"KD": [f"Animal {i}" for i in range(1, 5)],
                      "Scramble": [f"Animal {i}" for i in range(5, 9)]})
    assert result.p_value > 0.05
    assert result.ci_low < result.difference < result.ci_high
    assert result.n_a == 4 and result.n_b == 4
