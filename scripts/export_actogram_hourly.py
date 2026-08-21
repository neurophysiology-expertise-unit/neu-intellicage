#!/usr/bin/env python3
"""Export matched hourly all-visit and correct-visit actogram data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    out["success_rate"] = (
        out["correct_conditioned_visits"] / out["conditioned_visits"]
    ).where(out["conditioned_visits"].gt(0))
    return out


def plot_success_rate_actograms(table: pd.DataFrame, output: Path) -> None:
    """Double-plot hourly success rates; missing denominators remain blank."""
    animals = list(table["animal"].drop_duplicates())
    dates = list(pd.to_datetime(table["date"]).dt.date.drop_duplicates())
    fig, axes = plt.subplots(
        len(animals), 1, figsize=(10, 2.2 * len(animals)), squeeze=False
    )
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("#d9edf7")
    image = None
    for ax, animal in zip(axes[:, 0], animals):
        frame = table[table["animal"].eq(animal)]
        matrix = frame.pivot(
            index="date", columns="clock_hour", values="success_rate"
        ).reindex(index=dates, columns=range(24)).to_numpy(float)
        following = np.vstack([matrix[1:], np.full((1, 24), np.nan)])
        doubled = np.concatenate([matrix, following], axis=1)
        image = ax.imshow(
            doubled, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1
        )
        ax.set(
            ylabel=animal,
            xticks=[0, 12, 24, 36, 47],
            xticklabels=["0", "12", "24", "36", "48"],
            yticks=range(len(dates)),
            yticklabels=[str(date) for date in dates],
        )
        ax.tick_params(axis="y", labelsize=7)
    axes[-1, 0].set_xlabel("Clock hour (double plotted)")
    fig.suptitle("Hourly place-learning success rate by mouse\nBlue = no conditioned visits")
    colorbar = fig.colorbar(image, ax=axes[:, 0].tolist(), pad=.02, fraction=.025)
    colorbar.set_label("Correct conditioned visits / conditioned visits")
    fig.subplots_adjust(left=.10, right=.87, top=.91, bottom=.10, hspace=.35)
    fig.savefig(output / "hourly_success_rate_actograms.png", dpi=180)
    plt.close(fig)


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
    table[["animal", "date", "clock_hour", "all_visits", "conditioned_visits",
           "correct_conditioned_visits", "success_rate"]].to_csv(
        output / "hourly_visits_and_success_rate_per_mouse.csv", index=False
    )
    table[["animal", "date", "clock_hour", "conditioned_visits",
           "correct_conditioned_visits", "success_rate"]].to_csv(
        output / "hourly_success_rate_per_mouse.csv", index=False
    )
    plot_success_rate_actograms(table, output)
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
        "success_rate_definition": (
            "correct_conditioned_visits / conditioned_visits; missing when the denominator is zero"
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
