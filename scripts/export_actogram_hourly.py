#!/usr/bin/env python3
"""Export matched hourly all-visit and correct-visit actogram data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from neu_intellicage.io import load_session
from neu_intellicage.metrics import add_time_fields


def actogram_hourly_table(session_path: str | Path) -> pd.DataFrame:
    """Return a complete animal x calendar-date x clock-hour count table."""
    session = load_session(session_path)
    visits = add_time_fields(session.visits)
    animals = session.animals["AnimalName"].drop_duplicates().sort_values()
    dates = pd.date_range(
        visits["Start"].min().normalize(), visits["Start"].max().normalize(), freq="D"
    ).date
    index = pd.MultiIndex.from_product(
        [animals, dates, range(24)], names=["animal", "date", "clock_hour"]
    )
    visits["date"] = visits["Start"].dt.date
    visits["clock_hour"] = visits["Start"].dt.hour
    visits["correct_int"] = visits["correct"].fillna(False).astype(int)
    counts = visits.groupby(
        ["AnimalName", "date", "clock_hour"], observed=True
    ).agg(
        all_visits=("VisitID", "size"),
        conditioned_visits=("conditioned", "sum"),
        correct_conditioned_visits=("correct_int", "sum"),
    ).rename_axis(index=["animal", "date", "clock_hour"])
    out = counts.reindex(index, fill_value=0).reset_index()
    columns = ["all_visits", "conditioned_visits", "correct_conditioned_visits"]
    out[columns] = out[columns].astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = actogram_hourly_table(args.session)
    table.to_csv(output / "actogram_hourly_all_and_correct_visits.csv", index=False)
    table[["animal", "date", "clock_hour", "all_visits"]].to_csv(
        output / "actogram_hourly_all_visits.csv", index=False
    )
    table[["animal", "date", "clock_hour", "correct_conditioned_visits"]].to_csv(
        output / "actogram_hourly_correct_conditioned_visits.csv", index=False
    )
    visits_path = Path(args.session) / "IntelliCage" / "Visits.txt"
    metadata = {
        "source_session": Path(args.session).name,
        "source_visits_sha256": hashlib.sha256(visits_path.read_bytes()).hexdigest(),
        "date_start": str(table["date"].min()),
        "date_end": str(table["date"].max()),
        "animal_tags_in_output": False,
        "correct_trial_definition": (
            "conditioned visit with PlaceError == 0; neutral and incorrect visits excluded"
        ),
        "double_plot_note": (
            "CSV stores each animal-date-hour once; a 48-hour actogram repeats the next date visually"
        ),
        "script": Path(__file__).name,
    }
    (output / "actogram_hourly.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
