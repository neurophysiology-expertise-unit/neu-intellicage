"""Tests for the spontaneous-activity battery.

The cases that matter here are the ones that produced wrong answers before the
module existed: partial days at a session boundary, a night split across two
calendar rows, and pooling sessions whose overall level differs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neu_intellicage import activity
from neu_intellicage.groups import (days_to_separation, select_headline_measures,
                                    session_interaction_p)

ANIMALS = [f"Animal {i}" for i in range(1, 9)]
GROUPS = {"KD": ANIMALS[:4], "Scr": ANIMALS[4:]}


def make_visits(start="2026-08-11 07:00", days=4, per_hour=2, animals=ANIMALS,
                dark_bias=None, seed=0):
    """A synthetic cage record on a regular grid, one visit block per hour.

    ``dark_bias`` maps an animal to the share of its visits that fall in the
    dark phase, so a known group difference can be planted and recovered.
    """
    rng = np.random.default_rng(seed)
    rows, visit_id = [], 0
    begin = pd.Timestamp(start)
    for animal in animals:
        for hour in range(24 * days):
            stamp = begin + pd.Timedelta(hours=hour)
            dark = stamp.hour >= 19 or stamp.hour < 7
            n = per_hour
            if dark_bias is not None:
                share = dark_bias[animal]
                n = max(1, int(round(per_hour * (2 * share if dark else 2 * (1 - share)))))
            for k in range(n):
                visit_id += 1
                opened = stamp + pd.Timedelta(minutes=3 * k)
                rows.append({
                    "VisitID": visit_id, "AnimalName": animal,
                    "Start": opened, "End": opened + pd.Timedelta(seconds=60),
                    "Corner": 1 + (visit_id + hour) % 4,
                    "CornerCondition": 1, "PlaceError": 0,
                    "LickNumber": float(rng.integers(0, 40)),
                    "PresenceNumber": 3.0, "PresenceDuration": 30.0,
                    "GroupName": "KD" if animal in GROUPS["KD"] else "Scr",
                })
    return pd.DataFrame(rows)


def test_zeitgeber_puts_the_night_in_one_block():
    visits = make_visits(days=2)
    x = activity.add_zeitgeber(visits, lights_on=7, lights_off=19)
    dark = x[x["phase"].eq("dark")]
    # Clock hours 19-23 and 0-6 are the night; in ZT they must be 12..23 with no gap.
    assert set(dark["Start"].dt.hour) == set(range(19, 24)) | set(range(0, 7))
    assert set(dark["zt_hour"]) == set(range(12, 24))
    assert x[x["phase"].eq("light")]["zt_hour"].max() == 11


def test_partial_end_days_are_excluded():
    # Start mid-afternoon and stop mid-morning, so the first and last ZT days
    # are both incomplete. Only the fully covered days may be used.
    visits = make_visits(start="2026-08-11 13:00", days=4)
    visits = visits[visits["Start"] < pd.Timestamp("2026-08-14 10:00")]
    days = activity.complete_days(visits, lights_on=7)
    # 11 Aug is entered at 13:00, six hours after that ZT day opened, and the
    # record stops before 14 Aug closes -- so only 12 and 13 Aug are complete.
    assert [str(d.date()) for d in days] == ["2026-08-12", "2026-08-13"]
    daily = activity.daily_measures(visits)
    assert daily["zt_day"].nunique() == 2


def test_every_animal_gets_every_day_even_when_silent():
    visits = make_visits(days=3)
    # Animal 8 stops after the first ZT day.
    cut = pd.Timestamp("2026-08-12 07:00")
    visits = visits[~(visits["AnimalName"].eq("Animal 8") & visits["Start"].ge(cut))]
    daily = activity.daily_measures(visits)
    counts = daily[daily["measure"].eq("visits_per_day") & daily["phase"].eq("all")]
    per_animal = counts.groupby("AnimalName")["zt_day"].nunique()
    assert set(per_animal.unique()) == {counts["zt_day"].nunique()}
    silent = counts[counts["AnimalName"].eq("Animal 8")].sort_values("zt_day")
    assert silent["value"].iloc[0] > 0 and np.isnan(silent["value"].iloc[-1])


def test_phase_rates_are_per_hour_of_that_phase_not_per_day():
    # A perfectly flat animal visits at the same rate day and night, so its
    # light and dark rates must match its whole-day rate rather than halve.
    visits = make_visits(days=3, per_hour=2)
    daily = activity.daily_measures(visits)
    rates = (daily[daily["measure"].eq("visits_per_day")]
             .groupby("phase")["value"].mean())
    assert rates["light"] == pytest.approx(rates["all"], rel=1e-6)
    assert rates["dark"] == pytest.approx(rates["all"], rel=1e-6)


def test_profile_recovers_a_planted_dark_phase_difference():
    bias = {a: (0.55 if a in GROUPS["KD"] else 0.70) for a in ANIMALS}
    # per_hour=10 makes the planted share exact: 11 dark + 9 light visits is 0.55.
    visits = make_visits(days=5, per_hour=10, dark_bias=bias)
    profile = activity.session_profile(visits)
    fractions = profile.set_index("AnimalName")["dark_phase_visit_fraction"]
    assert fractions[GROUPS["KD"]].max() < fractions[GROUPS["Scr"]].min()
    assert fractions[GROUPS["KD"]].mean() == pytest.approx(0.55, abs=0.03)


def test_pooling_centres_out_a_session_wide_level_shift():
    """Two sessions with the same group difference but very different levels.

    Pooling the raw values lets the busier session dominate; centring on the
    day's cohort mean must leave the between-group difference untouched.
    """
    low = {a: (0.55 if a in GROUPS["KD"] else 0.65) for a in ANIMALS}
    high = {a: (0.80 if a in GROUPS["KD"] else 0.90) for a in ANIMALS}
    a_daily = activity.daily_measures(make_visits(start="2026-08-11 07:00", days=3,
                                                 per_hour=10, dark_bias=low))
    b_daily = activity.daily_measures(make_visits(start="2026-08-20 07:00", days=9,
                                                 per_hour=10, dark_bias=high))
    pooled = activity.pool_sessions({"low": a_daily, "high": b_daily})
    wide = activity.pooled_wide(pooled).set_index("AnimalName")
    column = "visits_per_day_dark"
    kd, scr = wide.loc[GROUPS["KD"], column], wide.loc[GROUPS["Scr"], column]
    assert kd.max() < scr.min()
    # Centring is a per-day constant shift, so the cohort's pooled mean is ~0
    # and the two sessions contribute on the same scale despite 3 vs 9 days.
    assert wide[column].mean() == pytest.approx(0.0, abs=1e-9)


def test_pooling_uncentred_keeps_the_raw_scale():
    daily = activity.daily_measures(make_visits(days=3, per_hour=4))
    pooled = activity.pool_sessions({"only": daily}, centre=False)
    raw = pooled[pooled["measure"].eq("visits_per_day") & pooled["phase"].eq("all")]
    assert raw["value"].mean() == pytest.approx(24 * 4, rel=0.02)


def test_same_calendar_day_in_two_sessions_is_not_merged():
    """Two sessions may cover the same date; centring must not mix their means."""
    daily = activity.daily_measures(make_visits(days=3, per_hour=4))
    pooled = activity.pool_sessions({"a": daily, "b": daily})
    counts = pooled[pooled["measure"].eq("visits_per_day") & pooled["phase"].eq("all")]
    assert set(counts["n_days"]) == {2 * daily["zt_day"].nunique()}


def test_headline_selection_drops_a_restatement():
    profile = pd.DataFrame({
        "AnimalName": ANIMALS,
        "L5": [1.0, 1.2, 1.4, 1.1, 3.0, 3.2, 3.4, 3.1],
        "RA": [-1.0, -1.2, -1.4, -1.1, -3.0, -3.2, -3.4, -3.1],   # r = -1.00 with L5
        # correlates with L5 at about 0.5: a different fact, so it stays
        "visits_per_day_all": [10, 40, 9, 38, 12, 11, 42, 39],
    })
    scan = pd.DataFrame({"measure": ["L5", "RA", "visits_per_day_all"],
                         "p_value": [0.03, 0.03, 0.06]})
    assert abs(profile["L5"].corr(profile["visits_per_day_all"])) < 0.95
    out = select_headline_measures(scan, profile).set_index("measure")
    assert out.loc["L5", "headline"] and not out.loc["RA", "headline"]
    assert "L5" in out.loc["RA", "redundant_with"]
    assert out.loc["visits_per_day_all", "headline"]


def test_interaction_permutes_the_animal_label_not_the_session():
    # A difference of exactly +2 in both sessions: no interaction to find.
    rows = []
    for session, base in (("s1", 10.0), ("s2", 30.0)):
        for i, animal in enumerate(ANIMALS):
            offset = 0.0 if animal in GROUPS["KD"] else 2.0
            rows.append({"AnimalName": animal, "session": session,
                         "value": base + offset + 0.01 * i})
    per_session = pd.DataFrame(rows)
    result = session_interaction_p(per_session, "value", GROUPS)
    assert result["observed_spread"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_value"] == 1.0
    assert set(result["per_session_difference"]) == {"s1", "s2"}


def test_interaction_flags_a_difference_that_reverses():
    rows = []
    for session, sign in (("s1", +1.0), ("s2", -1.0)):
        for animal in ANIMALS:
            offset = 0.0 if animal in GROUPS["KD"] else sign * 5.0
            rows.append({"AnimalName": animal, "session": session, "value": 10.0 + offset})
    result = session_interaction_p(pd.DataFrame(rows), "value", GROUPS)
    assert result["observed_spread"] == pytest.approx(10.0)
    assert result["p_value"] == pytest.approx(2 / 70)


def test_interaction_needs_every_animal_in_every_session():
    rows = [{"AnimalName": a, "session": "s1", "value": 1.0} for a in ANIMALS]
    rows += [{"AnimalName": a, "session": "s2", "value": 2.0} for a in ANIMALS[:6]]
    result = session_interaction_p(pd.DataFrame(rows), "value", GROUPS)
    assert np.isnan(result["p_value"]) and "every animal" in result["note"]


def test_days_to_separation_grows_more_confident():
    bias = {a: (0.55 if a in GROUPS["KD"] else 0.68) for a in ANIMALS}
    daily = activity.daily_measures(make_visits(days=6, per_hour=10, dark_bias=bias))
    curve = days_to_separation(daily, "visits_per_day", "dark", GROUPS)
    assert list(curve["days"]) == list(range(1, daily["zt_day"].nunique() + 1))
    assert curve["separated"].iloc[-1]
    assert curve["p_value"].iloc[-1] == pytest.approx(2 / 70)


def test_nosepokes_missing_table_counts_as_zero_not_missing():
    class Bare:
        visits = make_visits(days=2)
        nosepokes = pd.DataFrame()
    joined = activity.visits_with_nosepokes(Bare())
    assert (joined["nosepokes"] == 0).all()
    profile = activity.session_profile(Bare())
    assert (profile["nosepoke_probability_all"] == 0).all()


def test_describe_and_domain_cover_every_phase_column():
    visits = make_visits(days=3)
    profile = activity.session_profile(visits)
    for column in activity.measure_columns(profile):
        assert activity.domain_of(column) != "Other", column
        assert activity.describe(column) != column, column
