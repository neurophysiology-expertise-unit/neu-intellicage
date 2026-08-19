from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Session:
    path: Path
    animals: pd.DataFrame
    visits: pd.DataFrame
    nosepokes: pd.DataFrame
    hardware_events: pd.DataFrame


def _read(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required IntelliCage table not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype_backend="numpy_nullable")


def _dates(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def load_session(path: str | Path) -> Session:
    """Load one IntelliCage session without altering protocol-level fields."""
    root = Path(path).expanduser().resolve()
    cage = root / "IntelliCage"
    # Some IntelliCage exports append undocumented trailing fields to animal
    # rows without adding header names. Restrict parsing to the five documented
    # columns so pandas does not silently promote leading fields to an index.
    animals = pd.read_csv(
        root / "Animals.txt", sep="\t", usecols=range(5),
        names=["AnimalName", "AnimalTag", "Sex", "GroupName", "AnimalNotes"],
        header=0, dtype_backend="numpy_nullable",
    )
    visits = _dates(_read(cage / "Visits.txt"), ("Start", "End"))
    nosepokes = _dates(_read(cage / "Nosepokes.txt", required=False), ("Start", "End"))
    hardware = _dates(_read(cage / "HardwareEvents.txt", required=False), ("DateTime",))
    # Every column the analysis code reads unconditionally, so a truncated
    # export fails here with the column named rather than deep inside plotting.
    required = {"VisitID", "AnimalTag", "Start", "End", "Corner", "CornerCondition", "PlaceError"}
    missing = required.difference(visits.columns)
    if missing:
        raise ValueError(f"Visits.txt lacks required columns: {sorted(missing)}")
    if visits["VisitID"].duplicated().any():
        raise ValueError("VisitID must be unique within a session")
    visits["AnimalTag"] = visits["AnimalTag"].astype("string")
    animals["AnimalTag"] = animals["AnimalTag"].astype("string")
    visits = visits.merge(
        animals[["AnimalTag", "AnimalName", "GroupName"]],
        on="AnimalTag", how="left", validate="many_to_one",
    )
    if visits["AnimalName"].isna().any():
        unknown = sorted(visits.loc[visits["AnimalName"].isna(), "AnimalTag"].unique())
        raise ValueError(f"Visits contain tags absent from Animals.txt: {unknown}")
    return Session(root, animals, visits, nosepokes, hardware)
