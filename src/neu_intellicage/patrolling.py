"""Patrolling (corner-sequence) analysis.

In a patrolling protocol the rewarded corner is not fixed: it advances around
the cage each time the animal gets it right. The 25 August Verstreken export
implements the clockwise form, 1 -> 2 -> 3 -> 4 -> 1, with the target advancing
ONLY on a correct visit; this was verified against the export rather than
assumed (100% of rewarded visits were clockwise from the animal's previous
rewarded corner, in all eight animals). Voikar et al. 2018 call this family
"patrolling" and report it as the task where hippocampal animals are most
clearly impaired, because it requires the animal to remember where it is in a
sequence rather than a single place.

**Why chance is 1/3 here and not 1/4.** The target is never the corner the
animal is currently standing in: after a correct visit the target advances away
from it, and after an incorrect visit the target is by definition somewhere the
animal is not. So a mouse that has learned nothing except "do not go back into
the corner I just left" already scores 1/3. Judging patrolling against a 25%
line therefore credits that trivial strategy as sequence learning. The primary
measure here is the proportion of MOVES that hit the target, against a chance of
1/3; the all-visits proportion against 1/4 is kept as a secondary column so the
difference between the two is visible rather than hidden in a choice of
denominator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import add_time_fields

PATROL_CHANCE = 1 / 3        # over moves; see module docstring
NAIVE_CHANCE = 0.25          # over all visits, including re-entries


def annotate_patrol(visits: pd.DataFrame) -> pd.DataFrame:
    """Per-visit patrol annotations: target, move/repeat, correctness, run length."""
    x = add_time_fields(visits).sort_values(["AnimalName", "Start"]).copy()
    x["correct_patrol"] = x["CornerCondition"].eq(1)
    x["previous_corner"] = x.groupby("AnimalName")["Corner"].shift()
    x["is_repeat"] = x["Corner"].eq(x["previous_corner"])
    x["is_move"] = x["previous_corner"].notna() & ~x["is_repeat"]
    # The target is reconstructed from the controller's own record rather than
    # simulated, so a change of protocol cannot silently invalidate it.
    # The target in force at a visit comes from the last correct visit STRICTLY
    # BEFORE it, so the correct corner has to be shifted out of its own row
    # before being carried forward.
    correct_corner = x["Corner"].where(x["correct_patrol"])
    last_correct = correct_corner.groupby(x["AnimalName"]).shift().groupby(x["AnimalName"]).ffill()
    x["expected_target"] = (last_correct % 4) + 1
    x["run_length"] = 0
    for animal, frame in x.groupby("AnimalName"):
        run, values = 0, []
        for hit in frame["correct_patrol"]:
            run = run + 1 if hit else 0
            values.append(run)
        x.loc[frame.index, "run_length"] = values
    return x


def patrol_daily(visits: pd.DataFrame) -> pd.DataFrame:
    """Per animal and day: hit rate over moves, over all visits, and run lengths."""
    x = annotate_patrol(visits)
    x = x[x["conditioned"]]
    if x.empty:
        return pd.DataFrame()
    rows = []
    for (animal, date), frame in x.groupby(["AnimalName", "date"]):
        moves = frame[frame["is_move"]]
        completed = frame[frame["correct_patrol"]]
        rows.append({
            "AnimalName": animal, "GroupName": frame["GroupName"].iloc[0], "date": date,
            "visits": len(frame),
            "moves": len(moves),
            "correct_moves": int(moves["correct_patrol"].sum()),
            "hit_rate_moves": float(moves["correct_patrol"].mean()) if len(moves) else np.nan,
            "hit_rate_all_visits": float(frame["correct_patrol"].mean()),
            "repeat_rate": float(frame["is_repeat"].mean()),
            "mean_run_length": float(completed.groupby(
                (~frame["correct_patrol"]).cumsum()).size().mean()) if len(completed) else 0.0,
            "longest_run": int(frame["run_length"].max()),
            "laps_completed": int(frame["run_length"].ge(4).sum()),
        })
    return pd.DataFrame(rows)


def patrol_cumulative(visits: pd.DataFrame) -> pd.DataFrame:
    """Cumulative record: correct moves against moves made, per animal.

    This is the measure to watch day to day. A daily hit rate on a handful of
    moves is mostly noise; the cumulative record accumulates evidence, so its
    SLOPE is the learning rate and its distance above the chance diagonal is the
    total evidence of learning so far. ``excess_over_chance`` is that distance in
    correct moves -- the number of hits beyond what a non-learning mouse making
    the same number of moves would have produced.
    """
    x = annotate_patrol(visits)
    x = x[x["conditioned"] & x["is_move"]].copy()
    if x.empty:
        return pd.DataFrame()
    x["move_number"] = x.groupby("AnimalName").cumcount() + 1
    x["cumulative_correct"] = x.groupby("AnimalName")["correct_patrol"].cumsum()
    x["cumulative_hit_rate"] = x["cumulative_correct"] / x["move_number"]
    x["expected_by_chance"] = x["move_number"] * PATROL_CHANCE
    x["excess_over_chance"] = x["cumulative_correct"] - x["expected_by_chance"]
    return x[["AnimalName", "GroupName", "date", "Start", "move_number",
              "correct_patrol", "cumulative_correct", "cumulative_hit_rate",
              "expected_by_chance", "excess_over_chance"]]


def detect_task(visits: pd.DataFrame) -> str:
    """Name the task from the export: 'patrolling', 'place', or 'unconditioned'.

    Read from the data because the protocol a session actually ran is the only
    thing that determines how it must be scored, and filenames have already been
    wrong about this once.
    """
    x = add_time_fields(visits)
    conditioned = x[x["conditioned"]]
    if conditioned.empty:
        return "unconditioned"
    rewarded = conditioned[conditioned["CornerCondition"].eq(1)]
    if rewarded.empty:
        return "unconditioned"
    per_animal = rewarded.groupby("AnimalName")["Corner"].nunique()
    if (per_animal <= 1).all():
        return "place"
    annotated = annotate_patrol(visits)
    hits = annotated[annotated["correct_patrol"] & annotated["expected_target"].notna()]
    if len(hits) and (hits["Corner"] == hits["expected_target"]).mean() > 0.95:
        return "patrolling"
    return "place"
