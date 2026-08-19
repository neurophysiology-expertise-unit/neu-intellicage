import pandas as pd

from neu_intellicage.metrics import daily_learning, visit_block_learning
from neu_intellicage.report import _target_by_day


def visits():
    return pd.DataFrame({
        "VisitID": range(6), "AnimalName": ["A"] * 4 + ["B"] * 2,
        "GroupName": ["Control"] * 6,
        "Start": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 02:00", "2026-01-02 01:00", "2026-01-02 02:00", "2026-01-01 03:00", "2026-01-01 04:00"]),
        "PlaceError": [0, 1, 0, 0, 1, 1], "Corner": [1, 2, 1, 1, 3, 4],
    })


def test_daily_accuracy_uses_place_error_zero_as_correct():
    result = daily_learning(visits())
    a = result[result.AnimalName.eq("A")]
    assert a["accuracy"].tolist() == [0.5, 1.0]


def test_visit_blocks_are_per_animal():
    result = visit_block_learning(visits(), block_size=2)
    assert result[result.AnimalName.eq("A")]["accuracy"].tolist() == [0.5, 1.0]
    assert result[result.AnimalName.eq("B")]["accuracy"].tolist() == [0.0]


def test_target_corner_comes_from_positive_condition():
    frame = visits()
    frame["CornerCondition"] = [-1, 1, 1, -1, -1, 1]
    result = _target_by_day(frame)
    animal_a = result[result.AnimalName.eq("A")]
    assert animal_a["target_corner"].tolist() == [2, 1]
