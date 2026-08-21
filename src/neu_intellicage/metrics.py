from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

CHANCE = 0.25  # four corners, one rewarded


def chance_boundary(n: int, p0: float = CHANCE, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided binomial boundaries on a proportion of ``n`` choices under ``p0``.

    Returns (lower, upper) proportions. A point above ``upper`` is above chance
    at ``alpha``; a point below ``lower`` is below chance, which in a four-corner
    task means active avoidance of the target rather than failure to learn.

    A flat 25% reference answers "where is chance?" but not "is this mouse above
    it?", and the answer depends entirely on how many choices the point rests on:
    with 20 visits chance alone reaches 45%, with 400 it does not pass 30%. The
    boundary therefore has to move with n, and drawing it makes the reader's eye
    do the test. This is exact rather than normal-approximate because n is small
    on partial days and the approximation is poor in the tail.
    """
    if n <= 0:
        return float("nan"), float("nan")
    tail = alpha / 2

    def survival(k: int) -> float:  # P(X >= k)
        return sum(comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(k, n + 1))

    # An unattainable boundary is NaN, not an out-of-range number: with 10 visits
    # no count is significantly BELOW 25%, and saying so is the honest answer.
    upper = next((k for k in range(n + 1) if survival(k) <= tail), None)
    lower = next((k for k in range(n, -1, -1) if 1 - survival(k + 1) <= tail), None)
    return (float("nan") if lower is None else lower / n,
            float("nan") if upper is None else upper / n)


def boundary_frame(counts: pd.Series, p0: float = CHANCE, alpha: float = 0.05) -> pd.DataFrame:
    """Boundaries for a series of denominators, computed once per distinct n."""
    unique = {int(n): chance_boundary(int(n), p0, alpha) for n in pd.unique(counts.dropna())}
    return pd.DataFrame({
        "n": counts.to_numpy(),
        "chance_lower": [unique[int(n)][0] if pd.notna(n) else np.nan for n in counts],
        "chance_upper": [unique[int(n)][1] if pd.notna(n) else np.nan for n in counts],
    }, index=counts.index)


def add_time_fields(visits: pd.DataFrame) -> pd.DataFrame:
    """Add date/hour fields and the conditioned-visit correctness flags.

    IntelliCage sets ``PlaceError == 0`` both for a visit to the rewarded corner
    and for every visit made while no corner condition is active. Scoring
    accuracy as ``PlaceError == 0`` therefore reports 1.0 for habituation and
    nose-poke sessions, where no corner was ever rewarded. ``conditioned``
    marks the visits for which a target existed (``CornerCondition != 0``) and
    ``correct`` is defined only on those visits, so accuracy means
    "of the visits that could be right, how many were" in every session.
    """
    out = visits.copy()
    out["date"] = out["Start"].dt.date
    out["hour"] = out["Start"].dt.hour
    if "CornerCondition" in out:
        out["conditioned"] = out["CornerCondition"].ne(0).fillna(False).astype(bool)
    else:
        out["conditioned"] = True
    correct = out["PlaceError"].eq(0)
    out["correct"] = correct.where(out["conditioned"], other=pd.NA).astype("boolean")
    return out


def corner_entropy(visits: pd.DataFrame) -> pd.DataFrame:
    counts = visits.groupby(["AnimalName", "Corner"]).size().rename("n").reset_index()
    counts["p"] = counts["n"] / counts.groupby("AnimalName")["n"].transform("sum")
    entropy = counts.groupby("AnimalName")["p"].apply(lambda p: -(p * np.log2(p)).sum())
    return entropy.rename("corner_entropy_bits").reset_index()


def circadian_metrics(visits: pd.DataFrame, bin_hours: int = 1) -> pd.DataFrame:
    """Compute descriptive IS, IV and RA from hourly visit counts."""
    rows = []
    for animal, frame in visits.groupby("AnimalName"):
        start = frame["Start"].min().floor(f"{bin_hours}h")
        end = frame["Start"].max().ceil(f"{bin_hours}h")
        index = pd.date_range(start, end, freq=f"{bin_hours}h", inclusive="left")
        x = frame.set_index("Start").resample(f"{bin_hours}h").size().reindex(index, fill_value=0).astype(float)
        mean = x.mean()
        denom = ((x - mean) ** 2).sum()
        by_clock = x.groupby(x.index.hour).mean()
        is_value = len(x) * ((by_clock - mean) ** 2).sum() / (24 * denom) if denom else np.nan
        iv_value = len(x) * np.diff(x.to_numpy()).dot(np.diff(x.to_numpy())) / ((len(x) - 1) * denom) if len(x) > 1 and denom else np.nan
        hourly = by_clock.reindex(range(24), fill_value=0).to_numpy()
        m10 = max(np.roll(hourly, -i)[:10].mean() for i in range(24))
        l5 = min(np.roll(hourly, -i)[:5].mean() for i in range(24))
        ra = (m10 - l5) / (m10 + l5) if m10 + l5 else np.nan
        rows.append({"AnimalName": animal, "IS": is_value, "IV": iv_value, "RA": ra, "n_hours": len(x)})
    return pd.DataFrame(rows)

def daily_learning(visits: pd.DataFrame) -> pd.DataFrame:
    """Daily visit counts and accuracy over conditioned visits only.

    ``visits`` counts every visit (the activity measure); ``conditioned_visits``
    is the accuracy denominator. ``accuracy`` is NA on animal-days with no
    conditioned visit rather than a spurious 1.0.
    """
    x = add_time_fields(visits)
    out = x.groupby(["AnimalName", "GroupName", "date"], dropna=False).agg(
        visits=("VisitID", "size"), conditioned_visits=("conditioned", "sum"),
        correct=("correct", "sum"), accuracy=("correct", "mean"),
    ).reset_index()
    out["accuracy"] = out["accuracy"].where(out["conditioned_visits"].gt(0))
    return out


def cumulative_drinking_learning(visits: pd.DataFrame, nosepokes: pd.DataFrame,
                                 phases: list[dict]) -> pd.DataFrame:
    """Trial-by-trial cumulative successes for explicitly declared phases.

    Following IntelliR, a drinking attempt is a visit containing at least one
    nose-poke. Only conditioned visits can be scored. Phase dates are supplied
    explicitly because a controller error or re-reversal must never be silently
    combined with the intended reversal phase.
    """
    columns = ["phase", "AnimalName", "GroupName", "attempt_number",
               "cumulative_successes", "success", "VisitID", "Start"]
    if nosepokes.empty or not phases:
        return pd.DataFrame(columns=columns)
    x = add_time_fields(visits)
    attempted = set(nosepokes["VisitID"].dropna().unique())
    x = x[x["conditioned"] & x["VisitID"].isin(attempted)].copy()
    rows = []
    for phase in phases:
        part = _phase_dates(x, phase.get("dates", []))
        part = part.sort_values(["AnimalName", "Start", "VisitID"]).copy()
        if part.empty:
            continue
        part["phase"] = phase["label"]
        part["attempt_number"] = part.groupby("AnimalName").cumcount() + 1
        part["success"] = part["correct"].astype(int)
        part["cumulative_successes"] = part.groupby("AnimalName")["success"].cumsum()
        rows.append(part[columns])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def _phase_dates(frame: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    wanted = {pd.Timestamp(date).date() for date in dates}
    return frame[frame["date"].isin(wanted)]


def visit_block_learning(visits: pd.DataFrame, block_size: int = 100) -> pd.DataFrame:
    """Accuracy in consecutive blocks of ``block_size`` conditioned visits.

    Blocks are numbered over conditioned visits, so a block is a fixed amount of
    evidence about choice rather than a fixed amount of time in the cage. The
    final block of a session is usually incomplete; ``visits`` records its real
    size so downstream code can require a minimum.
    """
    x = add_time_fields(visits).sort_values(["AnimalName", "Start"])
    x = x[x["conditioned"]]
    if x.empty:
        return pd.DataFrame(columns=["AnimalName", "GroupName", "block", "visits", "accuracy"])
    x["visit_number"] = x.groupby("AnimalName").cumcount() + 1
    x["block"] = (x["visit_number"] - 1) // block_size + 1
    return x.groupby(["AnimalName", "GroupName", "block"], dropna=False).agg(
        visits=("VisitID", "size"), accuracy=("correct", "mean")
    ).reset_index()


def trials_to_criterion(blocks: pd.DataFrame, threshold: float = 0.5, consecutive: int = 2,
                        block_size: int = 100, min_block_visits: int | None = None) -> pd.DataFrame:
    """Conditioned visits needed for ``consecutive`` blocks at or above ``threshold``.

    Only complete blocks count towards the criterion: a 3-visit trailing block
    that happens to read 0.67 is not evidence of learning. ``min_block_visits``
    defaults to the full ``block_size``. Blocks must also be adjacent in block
    number, so a gap cannot be mistaken for a run.
    """
    floor = block_size if min_block_visits is None else min_block_visits
    rows = []
    for animal, frame in blocks.sort_values("block").groupby("AnimalName"):
        frame = frame.reset_index(drop=True)
        qualifies = (frame["accuracy"].ge(threshold) & frame["visits"].ge(floor)).to_numpy()
        numbers = frame["block"].to_numpy()
        run, trial, previous = 0, pd.NA, None
        for position, block in enumerate(numbers):
            adjacent = previous is not None and block == previous + 1
            run = (run + 1 if adjacent else 1) if qualifies[position] else 0
            previous = block
            if run >= consecutive:
                trial = int(block * block_size)
                break
        rows.append({"AnimalName": animal, "trials_to_criterion": trial, "threshold": threshold,
                     "consecutive_blocks": consecutive, "min_block_visits": floor})
    return pd.DataFrame(rows)
