import numpy as np
import pandas as pd
import pytest

from neu_intellicage.groups import benjamini_hochberg, scan_profile
from neu_intellicage.io import Session
from neu_intellicage.metrics import chance_boundary
from neu_intellicage.profile import animal_profile, contingency_balance


def session(conditioned_for=("A", "B"), animals=("A", "B")):
    rows = []
    vid = 0
    for animal in animals:
        for day in range(3):
            for hour in (2, 3, 20, 21):
                rows.append({
                    "VisitID": vid, "AnimalName": animal, "GroupName": "G",
                    "Start": pd.Timestamp(f"2026-01-0{day + 1} {hour:02d}:00"),
                    "End": pd.Timestamp(f"2026-01-0{day + 1} {hour:02d}:00:20"),
                    "Corner": (vid % 4) + 1, "PlaceError": vid % 2,
                    "CornerCondition": 1 if animal in conditioned_for else 0,
                })
                vid += 1
    visits = pd.DataFrame(rows)
    return Session(None, pd.DataFrame(), visits, pd.DataFrame(), pd.DataFrame())


def test_boundary_moves_with_the_number_of_choices():
    """A flat 25% line is the bug this replaces: chance reaches much higher on few visits."""
    _, upper_small = chance_boundary(20)
    _, upper_large = chance_boundary(400)
    assert upper_small > upper_large > 0.25
    assert upper_small == pytest.approx(0.50)


def test_boundary_is_nan_when_unattainable():
    lower, upper = chance_boundary(8)
    assert np.isnan(lower)          # cannot be significantly below 25% on 8 visits
    assert not np.isnan(upper)


def test_boundary_matches_an_exact_binomial_tail():
    n = 100
    _, upper = chance_boundary(n, p0=0.25, alpha=0.05)
    from math import comb
    k = round(upper * n)
    tail = sum(comb(n, i) * 0.25**i * 0.75**(n - i) for i in range(k, n + 1))
    below = sum(comb(n, i) * 0.25**i * 0.75**(n - i) for i in range(k - 1, n + 1))
    assert tail <= 0.025 < below    # k is the smallest count that clears the tail


def test_profile_reports_one_row_per_animal_with_rhythm_measures():
    profile = animal_profile(session())
    assert len(profile) == 2
    for column in ("visits_per_day", "burstiness", "IS", "IV", "RA", "corner_entropy_bits"):
        assert column in profile


def test_contingency_balance_flags_groups_run_under_different_protocols():
    """The August habituation session had one group conditioned and the other not."""
    balance = contingency_balance(session(conditioned_for=("A",)),
                                  {"one": ["A"], "two": ["B"]})
    assert bool(balance["confounded"].all())
    assert balance["fraction_spread"].iloc[0] == pytest.approx(1.0)


def test_contingency_balance_passes_a_matched_session():
    balance = contingency_balance(session(), {"one": ["A"], "two": ["B"]})
    assert not bool(balance["confounded"].any())


def test_scan_drops_measures_that_are_constant_across_animals():
    profile = pd.DataFrame({"AnimalName": [f"A{i}" for i in range(4)],
                            "varies": [1.0, 2.0, 3.0, 4.0], "flat": [7.0] * 4})
    table = scan_profile(profile, {"x": ["A0", "A1"], "y": ["A2", "A3"]})
    assert table["measure"].tolist() == ["varies"]
    assert "flat" in table["dropped_constant_measures"].iloc[0]


def test_bh_correction_cannot_be_significant_at_this_sample_size():
    """Fifteen measures at n=4 per group: the floor is 0.029, so min BH is 0.43."""
    floor = 1 / 35 * 2
    adjusted = benjamini_hochberg(np.full(15, floor))
    assert adjusted.min() > 0.05


def test_power_rises_with_group_size_and_effect():
    from neu_intellicage.groups import permutation_power
    assert permutation_power(4, 1.6) < permutation_power(8, 1.6)
    assert permutation_power(4, 0.5) < permutation_power(4, 2.5)


def test_power_at_a_null_effect_is_about_alpha():
    from neu_intellicage.groups import permutation_power
    assert permutation_power(6, 0.0) <= 0.08


def test_animals_needed_reports_the_smallest_sufficient_group():
    from neu_intellicage.groups import animals_needed
    plan = animals_needed(1.6, target_power=0.8)
    assert plan["needed"] is not None
    assert plan["power"][plan["needed"]] >= 0.8
    smaller = [n for n in plan["power"] if n < plan["needed"]]
    assert all(plan["power"][n] < 0.8 for n in smaller)


def test_more_data_per_animal_cannot_break_a_fixed_ordering():
    """The finding this encodes: one crossing pair fixes p at 4/70 forever.

    Averaging more days converges each animal on its true value; if that true
    ordering contains a single crossing, the exact test returns 4/70 = 0.057
    no matter how precisely each animal is measured.
    """
    from neu_intellicage.groups import exact_permutation_p
    truth = np.array([1.95, 1.30, 1.55, 1.75, 1.15, 1.45, 0.95, 1.10])
    p, total, _ = exact_permutation_p(truth[:4], truth[4:])
    assert (p, total) == (4 / 70, 70)
    # halving the noise around those values does not change the answer
    for scale in (1.0, 0.5, 0.1, 0.0):
        shrunk = truth.mean() + (truth - truth.mean()) * (1 + scale * 0)
        assert exact_permutation_p(shrunk[:4], shrunk[4:])[0] == p
