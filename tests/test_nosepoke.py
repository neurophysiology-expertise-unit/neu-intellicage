from pathlib import Path

import pandas as pd

from neu_intellicage.io import Session
from neu_intellicage.plots import nosepoke_acquisition


def test_nosepoke_acquisition_counts_visits_not_repeated_pokes(tmp_path: Path):
    visits = pd.DataFrame({
        "VisitID": [1, 2, 3],
        "AnimalName": ["A", "A", "A"],
        "GroupName": ["Control", "Control", "Control"],
        "Start": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 02:00", "2026-01-02 01:00"]),
        "PlaceError": [0, 0, 0],
    })
    nosepokes = pd.DataFrame({"VisitID": [1, 1, 3]})
    session = Session(tmp_path, pd.DataFrame(), visits, nosepokes, pd.DataFrame())

    nosepoke_acquisition(session, tmp_path)

    result = pd.read_csv(tmp_path / "daily_nosepoke_acquisition.csv")
    assert result["total_nosepokes"].tolist() == [2, 1]
    assert result["proportion_visits_with_nosepoke"].tolist() == [0.5, 1.0]
    assert (tmp_path / "daily_nosepoke_acquisition.png").exists()
