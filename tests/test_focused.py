from pathlib import Path

import pandas as pd

from neu_intellicage.focused import correct_visit_actograms
from neu_intellicage.io import Session


def test_correct_actogram_excludes_incorrect_and_neutral_visits(tmp_path: Path):
    cage = tmp_path / "IntelliCage"
    cage.mkdir()
    pd.DataFrame({
        "DateTime": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 06:30", "2026-01-01 18:30"]),
        "Illumination": [0, 40, 0],
    }).to_csv(cage / "Environment.txt", sep="\t", index=False)
    visits = pd.DataFrame({
        "VisitID": [1, 2, 3, 4], "AnimalName": ["A1", "A1", "A2", "A2"],
        "GroupName": ["G1", "G1", "G2", "G2"],
        "Start": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 08:00",
                                 "2026-01-01 09:00", "2026-01-01 20:00"]),
        "End": pd.to_datetime(["2026-01-01 01:01", "2026-01-01 08:01",
                               "2026-01-01 09:01", "2026-01-01 20:01"]),
        "PlaceError": [0, 1, 0, 0], "CornerCondition": [1, 1, 0, 1], "Corner": [1, 2, 1, 2],
    })
    session = Session(tmp_path, pd.DataFrame(), visits, pd.DataFrame(), pd.DataFrame())
    correct_visit_actograms(session, tmp_path / "out", {"G1": ["A1"], "G2": ["A2"]})
    summary = pd.read_csv(tmp_path / "out" / "correct_visit_light_phase_by_animal.csv")
    a1 = summary.set_index("AnimalName").loc["A1"]
    a2 = summary.set_index("AnimalName").loc["A2"]
    assert a1["correct_visits_dark"] == 1
    assert a1["correct_visits_light"] == 0
    assert a2["correct_visits_dark"] == 1
    assert a2["correct_visits_light"] == 0
    assert a1["all_visits_dark"] == 1
    assert a1["all_visits_light"] == 1
    assert a2["all_visits_dark"] == 1
    assert a2["all_visits_light"] == 1
    assert (tmp_path / "out" / "correct_visit_actograms.png").exists()
    assert (tmp_path / "out" / "all_visit_light_dark_rates.png").exists()
    assert (tmp_path / "out" / "correct_visit_light_phase_comparison.csv").exists()
