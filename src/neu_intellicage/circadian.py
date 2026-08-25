"""Circadian description of visit activity.

Adapted from `neu-oldenlabs`, which computes the same quantities on home-cage
distance traces. The measures are the standard non-parametric set (IS, IV, RA)
plus a cosinor fit; what IntelliCage contributes instead of distance is a visit
count per hour.

The cosinor is worth having alongside IS/IV/RA because it names WHEN the animal
is active. Acrophase is a clock hour, so a group difference in it is a phase
shift and reads directly as a biological statement; a difference in IV says only
that one group's day is more fragmented, without saying when.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import add_time_fields


def hourly_visits(visits: pd.DataFrame) -> pd.DataFrame:
    """Visits per animal per hour on a regular grid, zeros included.

    The grid matters: dropping empty hours would make a mouse that goes quiet
    for six hours look identical to one that was not recorded, and every
    rhythm measure below is a statement about the shape of that grid.
    """
    x = add_time_fields(visits)
    frames = []
    for animal, frame in x.groupby("AnimalName"):
        start = frame["Start"].min().floor("h")
        end = frame["Start"].max().ceil("h")
        index = pd.date_range(start, end, freq="h", inclusive="left")
        counts = frame.set_index("Start").resample("h").size().reindex(index, fill_value=0)
        frames.append(pd.DataFrame({"AnimalName": animal, "hour_start": index,
                                    "visits": counts.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def cosinor(series: pd.Series) -> dict:
    """Least-squares fit of mesor + amplitude * cos(2*pi*(t - acrophase)/24)."""
    values = series.to_numpy(dtype=float)
    hours = series.index.hour.to_numpy(dtype=float) + series.index.minute.to_numpy() / 60.0
    radians = 2 * np.pi * hours / 24.0
    design = np.column_stack([np.ones_like(radians), np.cos(radians), np.sin(radians)])
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return {"mesor": np.nan, "amplitude": np.nan, "acrophase_hour": np.nan}
    coefficients, *_ = np.linalg.lstsq(design[finite], values[finite], rcond=None)
    mesor, cosine, sine = coefficients
    return {"mesor": float(mesor), "amplitude": float(np.hypot(cosine, sine)),
            "acrophase_hour": float(np.arctan2(sine, cosine) * 24 / (2 * np.pi)) % 24}


def relative_amplitude(series: pd.Series) -> dict:
    """M10, L5 and RA from the mean 24-hour profile, treated as circular.

    The 10-hour and 5-hour windows are taken over the mean daily profile rather
    than the raw trace, so the result does not drift with recording length, and
    the profile is concatenated with itself before rolling so a window may wrap
    past midnight.
    """
    profile = series.groupby(series.index.hour).mean().reindex(range(24)).fillna(0)
    extended = pd.concat([profile, profile], ignore_index=True)
    m10 = float(extended.rolling(10, min_periods=10).mean().iloc[9:33].max())
    l5 = float(extended.rolling(5, min_periods=5).mean().iloc[4:28].min())
    total = m10 + l5
    return {"M10": m10, "L5": l5, "RA": float((m10 - l5) / total) if total else np.nan}


def circadian_profile(visits: pd.DataFrame) -> pd.DataFrame:
    """One row per animal: cosinor parameters, IS, IV, M10/L5/RA."""
    hourly = hourly_visits(visits)
    rows = []
    for animal, frame in hourly.groupby("AnimalName"):
        series = frame.set_index("hour_start")["visits"].astype(float)
        row = {"AnimalName": animal}
        row.update(cosinor(series))
        row.update(relative_amplitude(series))
        mean = series.mean()
        denominator = ((series - mean) ** 2).sum()
        by_clock = series.groupby(series.index.hour).mean()
        row["IS"] = (len(series) * ((by_clock - mean) ** 2).sum() / (24 * denominator)
                     if denominator else np.nan)
        difference = np.diff(series.to_numpy())
        row["IV"] = (len(series) * difference.dot(difference) / ((len(series) - 1) * denominator)
                     if len(series) > 1 and denominator else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def hourly_profile_by_animal(visits: pd.DataFrame) -> pd.DataFrame:
    """One row per animal, one column per clock hour: mean visits in that hour.

    This is the input to the hour-by-hour cluster test. Averaging across days
    first means every animal contributes one 24-point curve, so the test compares
    animals rather than animal-days.
    """
    hourly = hourly_visits(visits)
    hourly["hour"] = hourly["hour_start"].dt.hour
    wide = hourly.pivot_table(index="AnimalName", columns="hour", values="visits", aggfunc="mean")
    return wide.reindex(columns=range(24)).reset_index()


CIRCULAR_MEASURES = ("acrophase_hour",)


def circular_mean_hour(hours: np.ndarray) -> float:
    """Mean of clock hours on the circle.

    A plain mean of acrophases is not merely imprecise, it can be exactly wrong:
    eight mice peaking between 22:56 and 00:36 average to 11:51 arithmetically --
    midday, the opposite of the truth -- because the values straddle midnight.
    Averaging the unit vectors instead gives 23:42.
    """
    hours = np.asarray(hours, dtype=float)
    hours = hours[np.isfinite(hours)]
    if hours.size == 0:
        return float("nan")
    radians = 2 * np.pi * hours / 24
    return float(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()) * 24 / (2 * np.pi)) % 24


def circular_difference_hours(a: float, b: float) -> float:
    """Signed difference a - b in hours, wrapped to (-12, 12]."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float((a - b + 12) % 24 - 12)


def compare_phase(profile: pd.DataFrame, groups: dict, measure: str = "acrophase_hour") -> dict:
    """Exact permutation test on a difference of CIRCULAR means.

    The test statistic is the wrapped difference between the two groups' circular
    means, so it is valid across midnight where a difference of ordinary means is
    not. The null is the same exhaustive relabelling used everywhere else.
    """
    from itertools import combinations

    (name_a, members_a), (name_b, members_b) = list(groups.items())
    series = profile.set_index("AnimalName")[measure].astype(float)
    a = series.reindex([m for m in members_a if m in series.index]).dropna().to_numpy()
    b = series.reindex([m for m in members_b if m in series.index]).dropna().to_numpy()
    pooled = np.concatenate([a, b])
    observed = abs(circular_difference_hours(circular_mean_hour(a), circular_mean_hour(b)))
    extreme = total = 0
    for pick in combinations(range(len(pooled)), len(a)):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(pick)] = True
        candidate = abs(circular_difference_hours(circular_mean_hour(pooled[mask]),
                                                  circular_mean_hour(pooled[~mask])))
        total += 1
        extreme += candidate >= observed - 1e-12
    return {"measure": measure, "group_a": name_a, "group_b": name_b,
            "circular_mean_a": circular_mean_hour(a), "circular_mean_b": circular_mean_hour(b),
            "difference_hours": circular_difference_hours(circular_mean_hour(a),
                                                          circular_mean_hour(b)),
            "p_value": extreme / total, "permutations": total,
            "n_a": len(a), "n_b": len(b)}
