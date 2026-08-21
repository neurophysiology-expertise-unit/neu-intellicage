import numpy as np
import pandas as pd

from neu_intellicage.metrics import (cumulative_drinking_learning, daily_learning,
                                     trials_to_criterion, visit_block_learning)
from neu_intellicage.report import _target_by_day


def visits(corner_condition=None):
    frame = pd.DataFrame({
        "VisitID": range(6), "AnimalName": ["A"] * 4 + ["B"] * 2,
        "GroupName": ["Control"] * 6,
        "Start": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 02:00", "2026-01-02 01:00", "2026-01-02 02:00", "2026-01-01 03:00", "2026-01-01 04:00"]),
        "PlaceError": [0, 1, 0, 0, 1, 1], "Corner": [1, 2, 1, 1, 3, 4],
    })
    frame["CornerCondition"] = corner_condition if corner_condition is not None else [1, -1, 1, 1, -1, -1]
    return frame


def test_daily_accuracy_uses_place_error_zero_as_correct():
    result = daily_learning(visits())
    a = result[result.AnimalName.eq("A")]
    assert a["accuracy"].tolist() == [0.5, 1.0]


def test_unconditioned_visits_are_not_scored_as_correct():
    """A session with no corner condition has no accuracy, not accuracy 1.0."""
    result = daily_learning(visits(corner_condition=[0] * 6))
    assert result["accuracy"].isna().all()
    assert result["conditioned_visits"].tolist() == [0, 0, 0]
    assert result["visits"].tolist() == [2, 2, 2]


def test_accuracy_denominator_excludes_neutral_visits_in_a_mixed_session():
    """Animal A: one rewarded, one error, two neutral -> 0.5, not 0.75."""
    frame = visits(corner_condition=[1, -1, 0, 0, -1, -1])
    frame["Start"] = pd.to_datetime(["2026-01-01 01:00"] * 4 + ["2026-01-01 03:00", "2026-01-01 04:00"])
    result = daily_learning(frame)
    a = result[result.AnimalName.eq("A")].iloc[0]
    assert a["visits"] == 4 and a["conditioned_visits"] == 2
    assert a["accuracy"] == 0.5


def test_visit_blocks_are_per_animal():
    result = visit_block_learning(visits(), block_size=2)
    assert result[result.AnimalName.eq("A")]["accuracy"].tolist() == [0.5, 1.0]
    assert result[result.AnimalName.eq("B")]["accuracy"].tolist() == [0.0]


def test_target_corner_comes_from_positive_condition():
    frame = visits(corner_condition=[-1, 1, 1, -1, -1, 1])
    result = _target_by_day(frame)
    animal_a = result[result.AnimalName.eq("A")]
    assert animal_a["target_corner"].tolist() == [2, 1]


def test_tied_target_corners_yield_one_row_and_are_flagged():
    """A switch day with a quiet mouse ties; the pivot downstream needs one row."""
    frame = pd.DataFrame({
        "VisitID": [1, 2], "AnimalName": ["A", "A"], "GroupName": ["Control"] * 2,
        "Start": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 02:00"]),
        "PlaceError": [0, 0], "Corner": [3, 1], "CornerCondition": [1, 1],
    })
    result = _target_by_day(frame)
    assert len(result) == 1
    assert bool(result["ambiguous_target"].iloc[0]) is True
    assert result["target_corner"].iloc[0] == 1
    result.pivot(index="date", columns="AnimalName", values="target_corner")


def test_criterion_ignores_incomplete_blocks():
    """A 3-visit trailing block reading 1.0 is not two blocks of evidence."""
    blocks = pd.DataFrame({"AnimalName": ["A"] * 3, "GroupName": ["G"] * 3, "block": [1, 2, 3],
                           "visits": [100, 100, 3], "accuracy": [0.2, 0.6, 1.0]})
    assert pd.isna(trials_to_criterion(blocks, block_size=100)["trials_to_criterion"].iloc[0])


def test_criterion_requires_adjacent_blocks():
    blocks = pd.DataFrame({"AnimalName": ["A"] * 3, "GroupName": ["G"] * 3, "block": [1, 7, 8],
                           "visits": [100, 100, 100], "accuracy": [0.9, 0.9, 0.1]})
    assert pd.isna(trials_to_criterion(blocks, block_size=100)["trials_to_criterion"].iloc[0])
    blocks.loc[2, "accuracy"] = 0.9
    assert trials_to_criterion(blocks, block_size=100)["trials_to_criterion"].iloc[0] == 800


def test_cumulative_learning_uses_nosepoke_visits_and_resets_by_phase():
    frame = visits(corner_condition=[1, -1, 1, 1, -1, -1])
    pokes = pd.DataFrame({"VisitID": [0, 1, 1, 2, 3, 4]})
    phases = [{"label": "day 1", "dates": ["2026-01-01"]},
              {"label": "day 2", "dates": ["2026-01-02"]}]
    result = cumulative_drinking_learning(frame, pokes, phases)
    a1 = result[(result.AnimalName.eq("A")) & result.phase.eq("day 1")]
    a2 = result[(result.AnimalName.eq("A")) & result.phase.eq("day 2")]
    assert a1["attempt_number"].tolist() == [1, 2]
    assert a1["cumulative_successes"].tolist() == [1, 1]
    assert a2["attempt_number"].tolist() == [1, 2]
    assert a2["cumulative_successes"].tolist() == [1, 2]
