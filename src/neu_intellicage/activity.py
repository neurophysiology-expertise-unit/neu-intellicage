"""Spontaneous-activity phenotyping, kept separate from task performance.

Why a separate module and a separate report. Voikar et al. (2018) separated
hippocampal from prefrontal lesions using SPONTANEOUS behaviour -- visit rate,
the regularity of activity, its synchronisation with the light cycle -- rather
than learning scores. That logic is the reason this battery exists: it does not
require the knockdown to survive a long training protocol, and it does not stop
being measurable when the contingency changes or the hardware fails.

Three things distinguish this from `profile.animal_profile`, which describes one
session on its own terms:

1.  **Zeitgeber time.** A day here runs from lights-on to lights-on, not
    midnight to midnight. With a 19:00-07:00 dark phase, a calendar day cuts the
    night in half and puts the two halves at opposite ends of the plot; every
    rest-phase measure then straddles a row boundary. `add_zeitgeber` moves the
    origin so a night is one contiguous block.

2.  **Complete days only.** A ZT day is used only if the whole 24 hours lies
    inside the recorded span. A session's first and last days are almost always
    partial, and a partial day is not a quiet day -- averaging the two together
    silently biases every daily rate downwards and, worse, rotates the estimated
    phase. That artifact is not hypothetical: on the patrolling session, partial
    end days alone produced an apparent -0.2 h/day acrophase drift.

3.  **Phase splitting.** Every measure that can be computed on a subset of
    visits is computed three times -- whole day, light phase, dark phase. A
    difference that is invisible in the daily total is often plain in one phase,
    which is the point of running the battery rather than a single number.

Measures whose definition already spans the 24-hour cycle -- IS, IV, RA, M10,
L5, and the cosinor terms -- are NOT phase split, because a light-phase-only IS
is not a weaker version of IS, it is a different quantity with no accepted
meaning. They come from `circadian` unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import add_time_fields

LIGHTS_ON = 7
LIGHTS_OFF = 19

#: Measures computed on any subset of visits, and therefore split by phase.
PHASE_MEASURES = (
    "visits_per_day",
    "visit_duration_median_s",
    "time_in_corners_min_per_day",
    "nosepokes_per_visit",
    "nosepoke_probability",
    "licks_per_visit",
    "presence_events_per_visit",
    "presence_time_ratio",
    "corner_switch_rate",
    "corners_used",
    "corner_entropy_bits",
    "active_hours",
    "ivi_median_min",
    "burstiness",
)

#: Measures defined over the whole 24-hour cycle; splitting them is meaningless.
WHOLE_DAY_MEASURES = ("IS", "IV", "RA", "M10", "L5", "mesor", "amplitude",
                      "acrophase_hour", "dark_phase_visit_fraction")

PHASES = ("all", "light", "dark")

#: Behavioural domain of each measure, for grouping the report and the forest plot.
DOMAINS = {
    "visits_per_day": "Activity level",
    "time_in_corners_min_per_day": "Activity level",
    "active_hours": "Activity level",
    "visit_duration_median_s": "Visit structure",
    "nosepokes_per_visit": "Visit structure",
    "nosepoke_probability": "Visit structure",
    "licks_per_visit": "Visit structure",
    "presence_events_per_visit": "In-corner activity",
    "presence_time_ratio": "In-corner activity",
    "corner_switch_rate": "Exploration",
    "corners_used": "Exploration",
    "corner_entropy_bits": "Exploration",
    "ivi_median_min": "Temporal pattern",
    "burstiness": "Temporal pattern",
    "IV": "Temporal pattern",
    "IS": "Circadian",
    "RA": "Circadian",
    "M10": "Circadian",
    "L5": "Circadian",
    "mesor": "Circadian",
    "amplitude": "Circadian",
    "acrophase_hour": "Circadian",
    "dark_phase_visit_fraction": "Circadian",
}

NOTES = {
    "visits_per_day": "corner visits per ZT day (per phase: per 12 h of that phase)",
    "visit_duration_median_s": "median time inside a corner, seconds",
    "time_in_corners_min_per_day": "total time inside corners per ZT day, minutes",
    "nosepokes_per_visit": "mean nose-pokes per visit",
    "nosepoke_probability": "proportion of visits containing at least one nose-poke",
    "licks_per_visit": "mean licks per visit; 0 when the corner delivered no water",
    "presence_events_per_visit": "mean PresenceNumber per visit: times the presence sensor was "
                                 "triggered inside the corner, an in-corner restlessness proxy",
    "presence_time_ratio": "total PresenceDuration divided by total visit duration; a ratio of "
                           "sums, not a mean of per-visit ratios, because a 0.01 s visit gives a "
                           "ratio in the hundreds. It exceeds 1 when the presence sensor and the "
                           "visit clock disagree, which they do on 17% of visits",
    "IS": "interdaily stability: how reproducible the daily pattern is across days",
    "IV": "intradaily variability: how fragmented the day is",
    "RA": "relative amplitude of the rest-activity rhythm, (M10-L5)/(M10+L5)",
    "M10": "mean visits per hour over the most active 10 consecutive hours",
    "L5": "mean visits per hour over the least active 5 consecutive hours: the rest phase",
    "mesor": "cosinor rhythm-adjusted mean, visits per hour",
    "amplitude": "cosinor amplitude, visits per hour",
    "acrophase_hour": "cosinor peak as a clock hour; CIRCULAR, compared by circadian.compare_phase",
    "corner_switch_rate": "proportion of visits made to a corner other than the previous one",
    "corners_used": "distinct corners entered, max 4",
    "corner_entropy_bits": "Shannon entropy of corner use, max 2 bits",
    "active_hours": "distinct clock hours containing at least one visit",
    "ivi_median_min": "median inter-visit interval, minutes",
    "burstiness": "Goh-Barabasi burstiness of inter-visit intervals; +1 bursty, 0 Poisson, -1 regular",
    "dark_phase_visit_fraction": "proportion of visits falling in the dark phase",
}


def visits_with_nosepokes(source) -> pd.DataFrame:
    """Visits with a ``nosepokes`` count column joined from ``Nosepokes.txt``.

    Accepts either a `io.Session` or a plain visits frame. A session whose
    nosepoke table is missing or empty yields a count of 0 rather than NaN: the
    file being absent means no poke was recorded, which is a zero, whereas NaN
    would silently drop the animal from every poke measure.
    """
    visits = getattr(source, "visits", source)
    nosepokes = getattr(source, "nosepokes", None)
    out = visits.copy()
    if nosepokes is None or nosepokes.empty or "VisitID" not in nosepokes:
        out["nosepokes"] = 0.0
        return out
    counts = nosepokes.groupby("VisitID").size().rename("nosepokes")
    out = out.merge(counts, left_on="VisitID", right_index=True, how="left")
    out["nosepokes"] = out["nosepokes"].fillna(0).astype(float)
    return out


def add_zeitgeber(visits: pd.DataFrame, lights_on: int = LIGHTS_ON,
                  lights_off: int = LIGHTS_OFF) -> pd.DataFrame:
    """Add ``zt_day``, ``zt_hour`` and ``phase`` columns.

    ZT0 is lights-on, so ZT12-24 is the dark phase as one contiguous block.
    """
    if not 0 <= lights_on < 24 or not 0 <= lights_off < 24:
        raise ValueError(f"lights_on/lights_off must be clock hours, got {lights_on}/{lights_off}")
    out = add_time_fields(visits).copy()
    shifted = out["Start"] - pd.Timedelta(hours=lights_on)
    out["zt_day"] = shifted.dt.normalize()
    out["zt_hour"] = shifted.dt.hour
    day_length = (lights_off - lights_on) % 24
    out["phase"] = np.where(out["zt_hour"] < day_length, "light", "dark")
    return out


def complete_days(visits: pd.DataFrame, lights_on: int = LIGHTS_ON) -> list[pd.Timestamp]:
    """The ZT days whose full 24 hours lie inside the recorded span.

    The span is taken across the whole cohort, not per animal: a mouse that
    happens to make no visit in the first hour of a day has still been RECORDED
    for that hour, and dropping the day for that mouse alone would make the
    groups rest on different days.
    """
    x = add_zeitgeber(visits, lights_on)
    if x.empty:
        return []
    first, last = x["Start"].min(), x["Start"].max()
    offset = pd.Timedelta(hours=lights_on)
    days = sorted(x["zt_day"].unique())
    return [day for day in days
            if day + offset >= first and day + offset + pd.Timedelta(days=1) <= last]


def _burstiness(intervals: np.ndarray) -> float:
    """Goh & Barabasi burstiness of inter-event intervals, in [-1, 1]."""
    intervals = np.asarray(intervals, dtype=float)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(intervals) < 3:
        return float("nan")
    mean, sd = intervals.mean(), intervals.std(ddof=1)
    if mean + sd == 0:
        return float("nan")
    return float((sd - mean) / (sd + mean))


def _corner_entropy(corners: pd.Series) -> float:
    counts = corners.value_counts()
    if counts.empty:
        return float("nan")
    p = counts.to_numpy(dtype=float) / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _one_block(frame: pd.DataFrame, hours: float) -> dict:
    """Every phase-splittable measure for one animal, one ZT day, one phase.

    ``hours`` is the duration of the window the block covers, so that a rate is
    per unit of RECORDED time rather than per row. A light-phase block covers
    12 h, not 24, and dividing both by 24 would halve every phase rate.
    """
    n = len(frame)
    if n == 0:
        return {measure: float("nan") for measure in PHASE_MEASURES}
    ordered = frame.sort_values("Start")
    intervals = ordered["Start"].diff().dt.total_seconds().to_numpy() / 60.0
    duration = (ordered["End"] - ordered["Start"]).dt.total_seconds() if "End" in ordered else None
    switches = ordered["Corner"].ne(ordered["Corner"].shift())
    out = {
        "visits_per_day": n / (hours / 24.0),
        "visit_duration_median_s": float(duration.median()) if duration is not None else float("nan"),
        "time_in_corners_min_per_day": (float(duration.sum()) / 60.0 / (hours / 24.0)
                                        if duration is not None else float("nan")),
        "corner_switch_rate": float(switches.iloc[1:].mean()) if n > 1 else float("nan"),
        "corners_used": float(ordered["Corner"].nunique()),
        "corner_entropy_bits": _corner_entropy(ordered["Corner"]),
        "active_hours": float(ordered["Start"].dt.hour.nunique()),
        "ivi_median_min": float(np.nanmedian(intervals[1:])) if n > 1 else float("nan"),
        "burstiness": _burstiness(intervals[1:]),
    }
    out["nosepokes_per_visit"] = (float(ordered["nosepokes"].fillna(0).mean())
                                  if "nosepokes" in ordered else float("nan"))
    out["nosepoke_probability"] = (float(ordered["nosepokes"].fillna(0).gt(0).mean())
                                   if "nosepokes" in ordered else float("nan"))
    out["licks_per_visit"] = (float(ordered["LickNumber"].fillna(0).mean())
                              if "LickNumber" in ordered else float("nan"))
    out["presence_events_per_visit"] = (float(ordered["PresenceNumber"].fillna(0).mean())
                                        if "PresenceNumber" in ordered else float("nan"))
    if "PresenceDuration" in ordered and duration is not None:
        # Ratio of sums, not mean of ratios. Visit duration can be as short as
        # 0.01 s while PresenceDuration is measured on its own clock, so a
        # per-visit ratio reaches the hundreds and its mean is meaningless.
        total = float(duration.sum())
        presence = float(ordered["PresenceDuration"].fillna(0).sum())
        out["presence_time_ratio"] = presence / total if total > 0 else float("nan")
    else:
        out["presence_time_ratio"] = float("nan")
    return out


def daily_measures(source, lights_on: int = LIGHTS_ON,
                   lights_off: int = LIGHTS_OFF) -> pd.DataFrame:
    """Long table: one value per animal, complete ZT day, phase and measure.

    Every animal gets a row for every complete day and phase even when it made
    no visit at all, so a mouse that went quiet is a low value rather than a
    missing row that silently drops out of the mean.
    """
    visits = visits_with_nosepokes(source)
    x = add_zeitgeber(visits, lights_on, lights_off)
    days = complete_days(visits, lights_on)
    if not days:
        return pd.DataFrame(columns=["AnimalName", "zt_day", "phase", "measure", "value"])
    x = x[x["zt_day"].isin(days)]
    animals = sorted(add_time_fields(visits)["AnimalName"].unique())
    day_length = (lights_off - lights_on) % 24
    window = {"all": 24.0, "light": float(day_length), "dark": float(24 - day_length)}
    rows = []
    for animal in animals:
        per_animal = x[x["AnimalName"].eq(animal)]
        for day in days:
            per_day = per_animal[per_animal["zt_day"].eq(day)]
            for phase in PHASES:
                block = per_day if phase == "all" else per_day[per_day["phase"].eq(phase)]
                for measure, value in _one_block(block, window[phase]).items():
                    rows.append({"AnimalName": animal, "zt_day": day, "phase": phase,
                                 "measure": measure, "value": value})
            # A ratio between the two phases belongs to the day as a whole, so it
            # is emitted once under phase "all" rather than three times. It has to
            # live here and not only in `session_profile`, or it would be the one
            # measure that cannot be pooled across sessions.
            fraction = (float(per_day["phase"].eq("dark").mean()) if len(per_day)
                        else float("nan"))
            rows.append({"AnimalName": animal, "zt_day": day, "phase": "all",
                         "measure": "dark_phase_visit_fraction", "value": fraction})
    return pd.DataFrame(rows)


def session_profile(source, lights_on: int = LIGHTS_ON,
                    lights_off: int = LIGHTS_OFF) -> pd.DataFrame:
    """One row per animal: every phase measure averaged over complete ZT days,
    joined to the whole-day circadian measures.

    Averaging across days rather than pooling all visits gives every day equal
    weight, so one unusually busy day cannot dominate an animal's phenotype.
    """
    from .circadian import circadian_profile

    visits = visits_with_nosepokes(source)
    daily = daily_measures(visits, lights_on, lights_off)
    if daily.empty:
        return pd.DataFrame()
    daily = daily.assign(column=daily["measure"] + "_" + daily["phase"])
    wide = (daily.groupby(["AnimalName", "column"], as_index=False)["value"].mean()
            .pivot(index="AnimalName", columns="column", values="value"))
    wide.columns.name = None
    wide["n_days"] = daily.groupby("AnimalName")["zt_day"].nunique()

    x = add_zeitgeber(visits, lights_on, lights_off)
    days = complete_days(visits, lights_on)
    kept = x[x["zt_day"].isin(days)] if days else x
    dark_fraction = kept.groupby("AnimalName")["phase"].apply(lambda s: float(s.eq("dark").mean()))
    wide["dark_phase_visit_fraction"] = dark_fraction

    rhythm = circadian_profile(kept).set_index("AnimalName")
    for measure in ("IS", "IV", "RA", "M10", "L5", "mesor", "amplitude", "acrophase_hour"):
        if measure in rhythm:
            wide[measure] = rhythm[measure]
    return wide.reset_index()


def pool_sessions(daily_by_session: dict[str, pd.DataFrame],
                  centre: bool = True) -> pd.DataFrame:
    """Combine per-session daily tables into one per-animal value per measure.

    ``centre`` subtracts each DAY's cohort mean before averaging. That matters
    whenever the sessions differ in overall level -- and here they do: the same
    mice are 58% nocturnal under free adaptation and 83% nocturnal during place
    acquisition, because the contingency itself pushes activity into the dark.
    Pooling raw values would let that between-session shift, which is shared by
    both groups, dominate the between-group contrast and would make the pooled
    estimate depend on how many days of each session happened to be recorded.
    Centring removes exactly the part that is common to all animals on a day and
    leaves the between-animal contrast, which is what is being tested.

    Centring cannot manufacture a difference: it subtracts the same number from
    every animal on a given day, so the group difference on each day is
    unchanged. It only changes the WEIGHTING of days in the average.
    """
    frames = []
    for label, frame in daily_by_session.items():
        if frame.empty:
            continue
        frames.append(frame.assign(session=label))
    if not frames:
        return pd.DataFrame(columns=["AnimalName", "measure", "phase", "value", "n_days"])
    long = pd.concat(frames, ignore_index=True)
    # A day key must be unique across sessions: two sessions can cover the same
    # calendar day, and centring on a shared key would mix their cohort means.
    long["day_key"] = long["session"] + "|" + long["zt_day"].astype(str)
    long["raw"] = long["value"]
    if centre:
        cohort = long.groupby(["day_key", "measure", "phase"])["value"].transform("mean")
        long["value"] = long["value"] - cohort
    pooled = (long.groupby(["AnimalName", "measure", "phase"], as_index=False)
              .agg(value=("value", "mean"), raw=("raw", "mean"),
                   n_days=("day_key", "nunique")))
    return pooled


def pooled_wide(pooled: pd.DataFrame, column: str = "value") -> pd.DataFrame:
    """Pivot `pool_sessions` output to one row per animal, `measure_phase` columns.

    ``column`` selects the centred values (``"value"``, what the test uses) or
    the uncentred ones (``"raw"``, what a group mean should be QUOTED in --
    centred means are symmetric about zero by construction and are unreadable as
    levels). The group DIFFERENCE is identical either way, because centring
    subtracts the same number from every animal on a day.
    """
    if pooled.empty:
        return pd.DataFrame(columns=["AnimalName"])
    frame = pooled.assign(name=pooled["measure"] + "_" + pooled["phase"])
    wide = frame.pivot(index="AnimalName", columns="name", values=column)
    wide.columns.name = None
    return wide.reset_index()


def measure_columns(frame: pd.DataFrame) -> list[str]:
    """Every measure column in a profile, excluding identity and bookkeeping."""
    skip = {"AnimalName", "GroupName", "n_days", "session"}
    return [column for column in frame.columns if column not in skip]


def describe(column: str) -> str:
    """Human-readable note for a `measure_phase` column."""
    base, _, phase = column.rpartition("_")
    if base in NOTES and phase in PHASES:
        suffix = {"all": "whole ZT day", "light": "light phase only",
                  "dark": "dark phase only"}[phase]
        return f"{NOTES[base]} ({suffix})"
    return NOTES.get(column, column)


def domain_of(column: str) -> str:
    base, _, phase = column.rpartition("_")
    if phase in PHASES and base in DOMAINS:
        return DOMAINS[base]
    return DOMAINS.get(column, "Other")
