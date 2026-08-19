from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_inventory(sessions_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(sessions_dir).expanduser().resolve().iterdir()):
        visits_path = path / "IntelliCage" / "Visits.txt"
        animals_path = path / "Animals.txt"
        if not path.is_dir() or not animals_path.exists():
            continue
        animals = pd.read_csv(animals_path, sep="\t", usecols=range(5), header=0,
                              names=["AnimalName", "AnimalTag", "Sex", "GroupName", "AnimalNotes"],
                              dtype_backend="numpy_nullable")
        if visits_path.exists():
            visits = pd.read_csv(visits_path, sep="\t", usecols=["VisitID", "AnimalTag", "Start", "End", "CornerCondition"])
            rows.append({"session": path.name, "animals_registered": len(animals), "animals_with_visits": visits["AnimalTag"].nunique(),
                         "visits": len(visits), "start": visits["Start"].min() if len(visits) else None,
                         "end": visits["End"].max() if len(visits) else None,
                         "conditioned_visits": int(visits["CornerCondition"].ne(0).sum()) if len(visits) else 0})
        else:
            rows.append({"session": path.name, "animals_registered": len(animals), "animals_with_visits": 0, "visits": 0,
                         "start": None, "end": None, "conditioned_visits": 0})
    return pd.DataFrame(rows)
