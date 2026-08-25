import numpy as np
import pandas as pd
import pytest

from neu_intellicage.circadian import (circadian_profile, circular_difference_hours,
                                       circular_mean_hour, compare_phase, cosinor,
                                       hourly_profile_by_animal, relative_amplitude)
from neu_intellicage.groups import _find_clusters, cluster_permutation, hedges_g


def nocturnal_visits(peak_hour: int = 23, animals=("A", "B", "C", "D")):
    rows, vid = [], 0
    for animal in animals:
        for day in range(1, 5):
            for hour in range(24):
                # a clean cosine profile peaking at peak_hour
                n = int(round(6 + 5 * np.cos(2 * np.pi * (hour - peak_hour) / 24)))
                for _ in range(max(n, 0)):
                    rows.append({"VisitID": vid, "AnimalName": animal, "GroupName": "G",
                                 "Start": pd.Timestamp(f"2026-01-{day:02d} {hour:02d}:10"),
                                 "End": pd.Timestamp(f"2026-01-{day:02d} {hour:02d}:10:15"),
                                 "Corner": (vid % 4) + 1, "PlaceError": 0, "CornerCondition": 1})
                    vid += 1
    return pd.DataFrame(rows)


def test_circular_mean_survives_midnight():
    """The bug this exists for: these average to 11.85 arithmetically."""
    hours = np.array([0.33, 0.47, 0.60, 22.96, 23.26, 0.09, 23.16, 23.92])
    assert hours.mean() == pytest.approx(11.85, abs=0.1)   # the wrong answer
    assert circular_mean_hour(hours) == pytest.approx(23.85, abs=0.05)


def test_circular_difference_wraps():
    assert circular_difference_hours(0.5, 23.5) == pytest.approx(1.0)
    assert circular_difference_hours(23.5, 0.5) == pytest.approx(-1.0)


def test_cosinor_recovers_a_known_acrophase():
    profile = circadian_profile(nocturnal_visits(peak_hour=23))
    assert circular_mean_hour(profile["acrophase_hour"].to_numpy()) == pytest.approx(23, abs=1.0)
    assert (profile["amplitude"] > 0).all()


def test_relative_amplitude_windows_may_wrap_midnight():
    index = pd.date_range("2026-01-01", periods=48, freq="h")
    values = pd.Series([10.0 if h.hour >= 20 or h.hour < 4 else 1.0 for h in index], index=index)
    result = relative_amplitude(values)
    assert result["M10"] > result["L5"]
    assert 0 < result["RA"] <= 1


def test_hedges_g_applies_the_small_sample_correction():
    a, b = np.array([5.0, 6, 7, 8]), np.array([1.0, 2, 3, 4])
    pooled = np.sqrt((3 * a.var(ddof=1) + 3 * b.var(ddof=1)) / 6)
    assert hedges_g(a, b) == pytest.approx((a.mean() - b.mean()) / pooled * (1 - 3 / 23))


def test_a_wrapping_cluster_is_reported_once_not_twice():
    statistic = np.zeros(24)
    statistic[22:] = 3.0
    statistic[:2] = 3.0
    clusters = _find_clusters(statistic, 2.0)
    assert len(clusters) == 1
    assert sorted(clusters[0]["hours"]) == [0, 1, 22, 23]


def test_opposite_signed_runs_are_separate_clusters():
    statistic = np.zeros(24)
    statistic[5:8] = 3.0
    statistic[15:18] = -3.0
    assert len(_find_clusters(statistic, 2.0)) == 2


def test_cluster_permutation_finds_a_planted_phase_difference():
    early = nocturnal_visits(peak_hour=20, animals=("A", "B", "C", "D"))
    late = nocturnal_visits(peak_hour=2, animals=("E", "F", "G", "H"))
    hourly = hourly_profile_by_animal(pd.concat([early, late], ignore_index=True))
    result = cluster_permutation(hourly, {"early": list("ABCD"), "late": list("EFGH")})
    assert result["n_permutations"] == 70
    assert result["clusters"]
    assert min(c["p_value"] for c in result["clusters"]) <= 2 / 70 + 1e-9


def test_compare_phase_is_valid_across_midnight():
    a = nocturnal_visits(peak_hour=23, animals=("A", "B", "C", "D"))
    b = nocturnal_visits(peak_hour=1, animals=("E", "F", "G", "H"))
    profile = circadian_profile(pd.concat([a, b], ignore_index=True))
    result = compare_phase(profile, {"x": list("ABCD"), "y": list("EFGH")})
    assert abs(result["difference_hours"]) == pytest.approx(2.0, abs=0.6)
    assert result["permutations"] == 70
