from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import Session
from .metrics import add_time_fields, circadian_metrics, corner_entropy, daily_learning, trials_to_criterion, visit_block_learning


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def qc(session: Session, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    daily = daily_learning(session.visits)
    daily.to_csv(output / "visits_per_animal_day.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for animal, frame in daily.groupby("AnimalName"):
        ax.plot(pd.to_datetime(frame["date"]), frame["visits"], marker="o", label=animal)
    ax.set(ylabel="Visits", xlabel="Date", title="Visits per animal per day")
    ax.legend(frameon=False)
    _save(fig, output / "visits_per_animal_day.png")
    if not session.hardware_events.empty:
        cols = [c for c in ["HardwareType", "Cage", "Corner", "Side", "State"] if c in session.hardware_events]
        events = session.hardware_events.groupby(cols, dropna=False).size().rename("events").reset_index()
        events.to_csv(output / "hardware_event_counts.csv", index=False)


def nosepoke_acquisition(session: Session, output: Path) -> None:
    """Plot daily acquisition of nose-poking for each animal.

    The primary outcome is the proportion of corner visits containing at least
    one nose-poke. This measures whether the animal initiated the operant
    response without allowing repeated pokes during one visit to dominate the
    learning curve. Nose-pokes per visit are retained as a secondary column in
    the machine-readable table.
    """
    if session.nosepokes.empty:
        return
    output.mkdir(parents=True, exist_ok=True)
    visits = add_time_fields(session.visits)
    poke_counts = session.nosepokes.groupby("VisitID").size().rename("nosepokes")
    visits = visits.join(poke_counts, on="VisitID")
    visits["nosepokes"] = visits["nosepokes"].fillna(0).astype(int)
    visits["visit_with_nosepoke"] = visits["nosepokes"].gt(0)
    daily = visits.groupby(["AnimalName", "GroupName", "date"], dropna=False).agg(
        visits=("VisitID", "size"),
        visits_with_nosepoke=("visit_with_nosepoke", "sum"),
        total_nosepokes=("nosepokes", "sum"),
        proportion_visits_with_nosepoke=("visit_with_nosepoke", "mean"),
        nosepokes_per_visit=("nosepokes", "mean"),
    ).reset_index()
    daily.to_csv(output / "daily_nosepoke_acquisition.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for animal, frame in daily.groupby("AnimalName"):
        ax.plot(pd.to_datetime(frame["date"]), frame["proportion_visits_with_nosepoke"],
                marker="o", linewidth=1.8, label=animal)
    ax.set(xlabel="Date", ylabel="Visits containing ≥1 nose-poke",
           ylim=(0, 1), title="Individual daily nose-poke acquisition")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, output / "daily_nosepoke_acquisition.png")


def tier1(session: Session, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    x = add_time_fields(session.visits)
    # First calculate a rate for every actually observed animal-hour bin,
    # including bins with zero visits. Summing by clock hour overweights longer
    # sessions, while an incomplete pivot produces undefined SEM bands.
    hourly_bins = []
    for animal, frame in x.groupby("AnimalName"):
        start = frame["Start"].min().floor("h")
        end = frame["Start"].max().ceil("h")
        index = pd.date_range(start, end, freq="h", inclusive="left")
        counts = frame.set_index("Start").resample("h").size().reindex(index, fill_value=0)
        hourly_bins.append(pd.DataFrame({"AnimalName": animal, "time": index,
                                         "hour": index.hour, "visits": counts.to_numpy()}))
    hourly_bins = pd.concat(hourly_bins, ignore_index=True)
    hourly_bins.to_csv(output / "hourly_rate_by_animal.csv", index=False)
    hourly = hourly_bins.groupby(["AnimalName", "hour"])["visits"].mean().rename("visits_per_hour").reset_index()
    hourly.to_csv(output / "hourly_visits.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = hourly.pivot(index="hour", columns="AnimalName", values="visits_per_hour").reindex(range(24))
    mean, sem = pivot.mean(axis=1), pivot.sem(axis=1)
    ax.axvspan(19, 23, color="0.94", zorder=0, label="Nominal dark phase")
    ax.axvspan(0, 7, color="0.94", zorder=0)
    lower = (mean - sem).clip(lower=0)
    upper = mean + sem
    ax.fill_between(mean.index, lower, upper, color="tab:blue", alpha=.25,
                    edgecolor="tab:blue", linewidth=.8, zorder=2, label="SEM")
    ax.plot(mean.index, mean, color="tab:blue", linewidth=2, zorder=3, label="Animal mean")
    ax.set(xlim=(0, 23), ylim=(0, max(1, float(upper.max()) * 1.08)),
           xlabel="Hour of day", ylabel="Visits/hour (animal mean ± SEM)", title="Hourly activity")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _save(fig, output / "hourly_activity.png")
    # Double-plotted actogram: each row shows one day followed by the next.
    act = x.groupby(["AnimalName", "date", "hour"]).size().rename("visits").reset_index()
    animals = list(act["AnimalName"].unique())
    fig, axes = plt.subplots(len(animals), 1, figsize=(10, 2.2 * len(animals)), squeeze=False)
    matrices = {}
    for animal in animals:
        frame = act[act["AnimalName"].eq(animal)]
        matrix = frame.pivot(index="date", columns="hour", values="visits").reindex(
            columns=range(24), fill_value=0
        ).fillna(0)
        matrices[animal] = np.concatenate(
            [matrix.to_numpy(), np.roll(matrix.to_numpy(), -1, axis=0)], axis=1
        )
    common_max = max(float(matrix.max()) for matrix in matrices.values())
    image = None
    for ax, animal in zip(axes[:, 0], animals):
        image = ax.imshow(matrices[animal], aspect="auto", interpolation="nearest",
                          cmap="Greys", vmin=0, vmax=max(1, common_max))
        ax.set(ylabel=animal, xticks=[0, 12, 24, 36, 47], xticklabels=["0", "12", "24", "36", "48"])
    axes[-1, 0].set_xlabel("Zeitgeber/clock hour (double plotted)")
    fig.suptitle("Individual activity actograms")
    colorbar = fig.colorbar(image, ax=axes[:, 0].tolist(), pad=.02, fraction=.025)
    colorbar.set_label("Visits per hour")
    fig.subplots_adjust(left=.10, right=.87, top=.92, bottom=.10, hspace=.35)
    fig.savefig(output / "actograms.png", dpi=180)
    plt.close(fig)
    entropy = corner_entropy(x); entropy.to_csv(output / "corner_entropy.csv", index=False)
    circadian_metrics(x).to_csv(output / "circadian_metrics.csv", index=False)
    ivis = []
    for animal, frame in x.sort_values("Start").groupby("AnimalName"):
        delta = frame["Start"].diff().dt.total_seconds().div(60).dropna()
        ivis.extend({"AnimalName": animal, "interval_minutes": value} for value in delta if value > 0)
    ivi = pd.DataFrame(ivis); ivi.to_csv(output / "inter_visit_intervals.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for animal, frame in ivi.groupby("AnimalName"):
        ax.hist(frame["interval_minutes"], bins=np.logspace(-2, 4, 60), histtype="step", density=True, label=animal)
    ax.set_xscale("log"); ax.set(xlabel="Inter-visit interval (min)", ylabel="Density", title="Inter-visit intervals")
    ax.legend(frameon=False)
    _save(fig, output / "inter_visit_intervals.png")


def tier2(session: Session, output: Path, block_size: int = 100) -> None:
    output.mkdir(parents=True, exist_ok=True)
    daily = daily_learning(session.visits); daily.to_csv(output / "daily_learning.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for animal, frame in daily.groupby("AnimalName"):
        ax.plot(pd.to_datetime(frame["date"]), frame["accuracy"], marker="o", alpha=.55, lw=1, label=animal)
    cohort = daily.groupby("date")["accuracy"].agg(["mean", "sem"]).reset_index()
    dates = pd.to_datetime(cohort["date"])
    ax.plot(dates, cohort["mean"], color="black", lw=2.5, label="Cohort mean")
    ax.fill_between(dates, cohort["mean"]-cohort["sem"], cohort["mean"]+cohort["sem"], color="black", alpha=.15)
    ax.axhline(.25, ls="--", color="0.4", label="Four-corner chance")
    ax.set(xlabel="Date", ylabel="Correct-place visits / all visits", ylim=(0, 1), title="Place-learning acquisition")
    ax.legend(frameon=False, ncol=2)
    _save(fig, output / "daily_learning.png")
    blocks = visit_block_learning(session.visits, block_size); blocks.to_csv(output / "visit_block_learning.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for animal, frame in blocks.groupby("AnimalName"):
        ax.plot(frame["block"] * block_size, frame["accuracy"], marker="o", ms=3, label=animal)
    ax.axhline(.25, ls="--", color="0.4"); ax.set(xlabel="Visits completed", ylabel="Correct-place proportion", ylim=(0, 1), title="Visit-block acquisition")
    ax.legend(frameon=False); _save(fig, output / "visit_block_learning.png")
    criterion = trials_to_criterion(blocks, block_size=block_size)
    criterion.to_csv(output / "trials_to_criterion.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    valid = criterion.dropna(subset=["trials_to_criterion"])
    positions = np.arange(len(criterion))
    valid_positions = positions[criterion["trials_to_criterion"].notna().to_numpy()]
    ax.scatter(valid_positions, valid["trials_to_criterion"],
               c=[f"C{i}" for i in valid_positions], s=55)
    for position, row in criterion[criterion["trials_to_criterion"].isna()].iterrows():
        ax.text(position, .04, "not reached", rotation=90, ha="center", va="bottom",
                transform=ax.get_xaxis_transform(), fontsize=8, color="0.4")
    ax.set(xticks=positions, xticklabels=criterion["AnimalName"], ylabel="Trials to criterion",
           xlim=(-.5, len(criterion) - .5), title="Criterion: ≥50% for two blocks")
    _save(fig, output / "trials_to_criterion.png")
    terminal = blocks.groupby("AnimalName").tail(1); terminal.to_csv(output / "terminal_accuracy.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(terminal))
    ax.scatter(positions, terminal["accuracy"], c=[f"C{i}" for i in positions], s=60)
    ax.axhline(.25, ls="--", color="0.4")
    ax.set(xticks=positions, xticklabels=terminal["AnimalName"], ylabel="Correct-place proportion",
           xlim=(-.5, len(terminal) - .5), ylim=(0, 1), title="Terminal 100-visit block by mouse")
    _save(fig, output / "terminal_accuracy.png")
    x = add_time_fields(session.visits)
    errors = x.groupby(["AnimalName", "date"]).agg(place_error_rate=("PlaceError", "mean"), visits=("VisitID", "size"), accuracy=("correct", "mean")).reset_index()
    if "SideError" in session.nosepokes:
        side = session.nosepokes.merge(session.visits[["VisitID", "AnimalName", "Start"]], on="VisitID", suffixes=("", "_visit"))
        side["date"] = side["Start_visit"].dt.date
        side = side.groupby(["AnimalName", "date"])["SideError"].mean().rename("side_error_rate").reset_index()
        errors = errors.merge(side, on=["AnimalName", "date"], how="left")
    errors.to_csv(output / "activity_accuracy_errors.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    cohort_errors = errors.groupby("date").mean(numeric_only=True).reset_index()
    ax.plot(pd.to_datetime(cohort_errors["date"]), cohort_errors["place_error_rate"], marker="o", label="Place error")
    if "side_error_rate" in cohort_errors:
        ax.plot(pd.to_datetime(cohort_errors["date"]), cohort_errors["side_error_rate"], marker="o", label="Side error")
    ax.set(xlabel="Date", ylabel="Error proportion", ylim=(0, 1), title="Error decomposition")
    ax.legend(frameon=False); _save(fig, output / "error_decomposition.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    for animal, frame in errors.groupby("AnimalName"):
        ax.scatter(frame["visits"], frame["accuracy"], label=animal)
    ax.axhline(.25, ls="--", color="0.4")
    ax.set(xlabel="Visits per day", ylabel="Correct-place proportion", ylim=(0, 1), title="Activity–accuracy dissociation")
    ax.legend(frameon=False)
    _save(fig, output / "activity_accuracy.png")
