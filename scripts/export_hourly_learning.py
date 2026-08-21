#!/usr/bin/env python3
"""Export a share-ready per-animal hourly table for one learning window.

The output is processed numerical data, not a copy of the IntelliCage raw
tables. Animal transponder tags are deliberately omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from neu_intellicage.io import load_session
from neu_intellicage.metrics import add_time_fields


def hourly_learning_table(
    session_path: str | Path,
    start: str,
    end: str,
    groups: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return one row per animal and clock hour, including zero-count rows."""
    session = load_session(session_path)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be later than start")

    visits = add_time_fields(session.visits)
    visits = visits[visits["Start"].between(start_ts, end_ts, inclusive="both")].copy()
    if visits.empty:
        raise ValueError("No visits fall inside the requested window")
    if "LickNumber" not in visits:
        raise ValueError("Visits.txt lacks required column: LickNumber")

    first_hour = start_ts.floor("h")
    last_hour = end_ts.floor("h")
    hours = pd.date_range(first_hour, last_hour, freq="h")
    animals = session.animals["AnimalName"].drop_duplicates().sort_values()
    grid = pd.MultiIndex.from_product(
        [animals, hours], names=["animal", "hour_start_local"]
    ).to_frame(index=False)

    visits["hour_start_local"] = visits["Start"].dt.floor("h")
    visits["correct_int"] = visits["correct"].fillna(False).astype(int)
    hourly = visits.groupby(
        ["AnimalName", "hour_start_local"], observed=True
    ).agg(
        total_visits=("VisitID", "size"),
        total_licks=("LickNumber", "sum"),
        conditioned_visits=("conditioned", "sum"),
        correct_conditioned_visits=("correct_int", "sum"),
    ).reset_index().rename(columns={"AnimalName": "animal"})

    out = grid.merge(hourly, on=["animal", "hour_start_local"], how="left")
    count_columns = [
        "total_visits", "total_licks", "conditioned_visits",
        "correct_conditioned_visits",
    ]
    out[count_columns] = out[count_columns].fillna(0).astype(int)
    out["hour_end_local"] = out["hour_start_local"] + pd.Timedelta(hours=1)
    observed_start = out["hour_start_local"].where(
        out["hour_start_local"].ge(start_ts), start_ts
    )
    observed_end = out["hour_end_local"].where(
        out["hour_end_local"].le(end_ts), end_ts
    )
    out["recorded_minutes"] = (
        (observed_end - observed_start).dt.total_seconds() / 60
    ).round(3)
    out["complete_hour"] = out["recorded_minutes"].eq(60)
    out["success_rate"] = (
        out["correct_conditioned_visits"] / out["conditioned_visits"]
    ).where(out["conditioned_visits"].gt(0))
    out["treatment_group"] = out["animal"].map(groups or {}).fillna("not supplied")
    out["phase"] = "initial place learning before target switch"
    out["window_start_local"] = start_ts
    out["window_end_local"] = end_ts
    return out[[
        "animal", "treatment_group", "phase", "hour_start_local",
        "hour_end_local", "recorded_minutes", "complete_hour", "total_visits",
        "total_licks", "conditioned_visits", "correct_conditioned_visits",
        "success_rate", "window_start_local", "window_end_local",
    ]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tau-animals", nargs="*", default=[],
        help="Animal names assigned to the Tau KD group",
    )
    parser.add_argument(
        "--scramble-animals", nargs="*", default=[],
        help="Animal names assigned to the Scramble group",
    )
    args = parser.parse_args()
    groups = {animal: "Tau KD" for animal in args.tau_animals}
    groups.update({animal: "Scramble" for animal in args.scramble_animals})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly_learning_table(args.session, args.start, args.end, groups).to_csv(
        output, index=False, date_format="%Y-%m-%d %H:%M:%S.%f"
    )
    visits_path = Path(args.session) / "IntelliCage" / "Visits.txt"
    metadata = {
        "dataset": output.name,
        "source_session": Path(args.session).name,
        "source_visits_sha256": hashlib.sha256(visits_path.read_bytes()).hexdigest(),
        "window_start_local": args.start,
        "window_end_local": args.end,
        "success_rate_definition": (
            "correct_conditioned_visits / conditioned_visits; missing when the denominator is zero"
        ),
        "animal_tags_in_output": False,
        "script": Path(__file__).name,
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
