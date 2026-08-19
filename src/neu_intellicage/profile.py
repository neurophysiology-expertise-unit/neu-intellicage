"""A per-animal behavioural profile: one row per mouse, many measures.

The measures are chosen from what the IntelliCage literature says actually
separates groups, not from what is easy to compute.

Voikar et al. 2018 is the reason most of this panel is about SPONTANEOUS
behaviour rather than task accuracy: hippocampal and prefrontal lesions were
discriminated during free adaptation, before any conditioning, by the
organisation of activity — visit rate, its regularity, and its synchronisation
with the light cycle. A tau knockdown with a decaying time window may likewise
show itself in how a mouse organises its day long before it shows itself in a
learning curve.

The mobility block exists because of the motor confound: tau deficiency has been
reported to produce parkinsonism (Lei et al. 2012), and IntelliCage observes
everything through corner visits, so a motor change can masquerade as a
cognitive one. Activity and accuracy measures must therefore be reported side by
side and never collapsed into a single "performance" number.

Every measure here is descriptive. Nothing in this module decides significance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import add_time_fields, circadian_metrics, corner_entropy

DARK_START, DARK_END = 19, 7  # nominal; unverified against the room schedule


def _burstiness(intervals: np.ndarray) -> float:
    """Goh & Barabasi burstiness of the inter-visit intervals, in [-1, 1].

    +1 is maximally bursty (long silences broken by tight bouts), 0 is Poisson,
    -1 is perfectly regular. This is the quantitative form of the "regularity vs
    erraticness" contrast the analysis plan asks for, and unlike a raw
    coefficient of variation it is bounded and comparable across animals with
    very different visit rates.
    """
    if len(intervals) < 3:
        return np.nan
    mean, sd = intervals.mean(), intervals.std(ddof=1)
    return float((sd - mean) / (sd + mean)) if (sd + mean) else np.nan


def animal_profile(session, dates: list[str] | None = None) -> pd.DataFrame:
    """One row per animal with activity, mobility, rhythm and task measures."""
    x = add_time_fields(session.visits).sort_values(["AnimalName", "Start"])
    if dates:
        wanted = {pd.Timestamp(d).date() for d in dates}
        x = x[x["date"].isin(wanted)]
    if x.empty:
        return pd.DataFrame()
    x["duration_s"] = (x["End"] - x["Start"]).dt.total_seconds()
    x["is_dark"] = (x["hour"] >= DARK_START) | (x["hour"] < DARK_END)
    # load_session returns an empty frame when Nosepokes.txt is absent, so the
    # groupby has to be guarded: a session can legitimately have no pokes at all.
    if session.nosepokes.empty or "VisitID" not in session.nosepokes:
        x["nosepokes"] = 0.0
    else:
        pokes = session.nosepokes.groupby("VisitID").size().rename("nosepokes")
        x = x.join(pokes, on="VisitID")
        x["nosepokes"] = x["nosepokes"].fillna(0)
    x["with_poke"] = x["nosepokes"].gt(0)

    rows = []
    for animal, frame in x.groupby("AnimalName"):
        days = frame["date"].nunique()
        intervals = frame["Start"].diff().dt.total_seconds().div(60).dropna()
        intervals = intervals[intervals > 0].to_numpy()
        # A corner switch is a visit to a different corner than the previous one:
        # the cheapest available index of how much the mouse moves around the cage.
        switches = frame["Corner"].ne(frame["Corner"].shift()).iloc[1:]
        conditioned = frame[frame["conditioned"]]
        rows.append({
            "AnimalName": animal,
            "GroupName": frame["GroupName"].iloc[0],
            # -- activity --
            "visits_per_day": len(frame) / days if days else np.nan,
            "visit_duration_median_s": float(frame["duration_s"].median()),
            "total_time_in_corners_min": float(frame["duration_s"].sum() / 60),
            "nosepokes_per_visit": float(frame["nosepokes"].mean()),
            "nosepoke_probability": float(frame["with_poke"].mean()),
            # -- mobility / exploration --
            "corner_switch_rate": float(switches.mean()) if len(switches) else np.nan,
            "corners_used_per_day": float(frame.groupby("date")["Corner"].nunique().mean()),
            "active_hours_per_day": float(frame.groupby("date")["hour"].nunique().mean()),
            # -- temporal organisation --
            "ivi_median_min": float(np.median(intervals)) if len(intervals) else np.nan,
            "burstiness": _burstiness(intervals),
            "dark_phase_visit_fraction": float(frame["is_dark"].mean()),
            # -- task --
            "accuracy": float(conditioned["correct"].mean()) if len(conditioned) else np.nan,
            "conditioned_visits": int(frame["conditioned"].sum()),
        })
    profile = pd.DataFrame(rows)
    rhythm = circadian_metrics(x)[["AnimalName", "IS", "IV", "RA"]]
    profile = profile.merge(rhythm, on="AnimalName", how="left")
    profile = profile.merge(corner_entropy(x), on="AnimalName", how="left")
    return profile


MEASURE_BLOCKS = {
    "Activity": ["visits_per_day", "visit_duration_median_s", "total_time_in_corners_min",
                 "nosepokes_per_visit", "nosepoke_probability"],
    "Mobility": ["corner_switch_rate", "corners_used_per_day", "active_hours_per_day",
                 "corner_entropy_bits"],
    "Rhythm": ["ivi_median_min", "burstiness", "dark_phase_visit_fraction", "IS", "IV", "RA"],
    "Task": ["accuracy"],
}

MEASURE_NOTES = {
    "visits_per_day": "corner visits per recorded day",
    "visit_duration_median_s": "median time inside a corner, seconds",
    "total_time_in_corners_min": "total time inside corners, minutes",
    "nosepokes_per_visit": "mean nose-pokes per visit",
    "nosepoke_probability": "proportion of visits containing at least one nose-poke",
    "corner_switch_rate": "proportion of visits made to a different corner than the previous visit",
    "corners_used_per_day": "distinct corners entered per day, max 4",
    "active_hours_per_day": "distinct clock hours containing at least one visit, per day",
    "corner_entropy_bits": "Shannon entropy of the corner-use distribution, max 2 bits",
    "ivi_median_min": "median inter-visit interval, minutes",
    "burstiness": "Goh-Barabasi burstiness of inter-visit intervals; +1 bursty, 0 Poisson, -1 regular",
    "dark_phase_visit_fraction": "proportion of visits in the nominal dark phase, 19:00-07:00, UNVERIFIED",
    "IS": "interdaily stability; higher means a more reproducible daily pattern",
    "IV": "intradaily variability; higher means a more fragmented day",
    "RA": "relative amplitude of the rest-activity rhythm",
    "accuracy": "proportion of conditioned visits made to the target corner",
}


def contingency_balance(session, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Did the groups experience the same corner contingency in this session?

    A between-group difference means nothing if the groups were not run under
    the same protocol. In the August habituation session Animals 1-4 had a
    corner condition on 100% of visits while Animals 5-8 had one on 0%, so any
    activity difference there is as consistent with "a contingency fragments
    activity" as with any treatment effect. That confound is invisible in the
    result table, so it is computed and reported alongside it.
    """
    x = add_time_fields(session.visits)
    per_animal = x.groupby("AnimalName")["conditioned"].mean()
    rows = []
    for name, members in groups.items():
        held = per_animal.reindex([m for m in members if m in per_animal.index])
        rows.append({"group": name, "animals": len(held),
                     "conditioned_visit_fraction": float(held.mean()) if len(held) else np.nan})
    frame = pd.DataFrame(rows)
    spread = frame["conditioned_visit_fraction"].max() - frame["conditioned_visit_fraction"].min()
    frame["confounded"] = spread > 0.05
    frame["fraction_spread"] = spread
    return frame
