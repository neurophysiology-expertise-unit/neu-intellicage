from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import Session
from .metrics import (CHANCE, add_time_fields, boundary_frame, chance_boundary, circadian_metrics,
                      corner_entropy, cumulative_drinking_learning, daily_learning,
                      trials_to_criterion, visit_block_learning)


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


def cumulative_learning(session: Session, output: Path, phases: list[dict],
                        groups: dict[str, list[str]] | None = None) -> None:
    """Plot literature-style cumulative success curves by declared phase."""
    attempts = cumulative_drinking_learning(session.visits, session.nosepokes, phases)
    if attempts.empty:
        return
    output.mkdir(parents=True, exist_ok=True)
    animal_group = {animal: group for group, animals in (groups or {}).items()
                    for animal in animals}
    attempts["AnalysisGroup"] = attempts["AnimalName"].map(animal_group).fillna(
        attempts["GroupName"].astype(str))
    attempts.to_csv(output / "cumulative_drinking_attempts.csv", index=False)

    terminal = attempts.sort_values("attempt_number").groupby(
        ["phase", "AnalysisGroup", "AnimalName"], as_index=False).tail(1).copy()
    terminal["success_slope"] = terminal["cumulative_successes"] / terminal["attempt_number"]
    terminal[["phase", "AnalysisGroup", "AnimalName", "attempt_number",
              "cumulative_successes", "success_slope"]].to_csv(
                  output / "cumulative_learning_slopes_by_animal.csv", index=False)

    labels = [phase["label"] for phase in phases if phase["label"] in set(attempts["phase"])]
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)
    palette = {name: f"C{i}" for i, name in enumerate(sorted(attempts["AnalysisGroup"].unique()))}
    summary_rows = []
    rng = np.random.default_rng(20260820)
    for ax, label in zip(axes.flat, labels):
        phase_data = attempts[attempts["phase"].eq(label)]
        phase_terminal = terminal[terminal["phase"].eq(label)]
        max_attempt = int(phase_data["attempt_number"].max())
        xline = np.arange(max_attempt + 1)
        ax.plot(xline, CHANCE * xline, ls="--", color="0.35", lw=1.5,
                label="25% chance slope")
        for animal, frame in phase_data.groupby("AnimalName"):
            group = frame["AnalysisGroup"].iloc[0]
            ax.plot(frame["attempt_number"], frame["cumulative_successes"],
                    color=palette[group], alpha=.28, lw=1)
            last = frame.iloc[-1]
            ax.text(last["attempt_number"], last["cumulative_successes"],
                    animal.replace("Animal ", "A"), color=palette[group], fontsize=7,
                    alpha=.8, ha="left", va="center")
        for group, frame in phase_terminal.groupby("AnalysisGroup"):
            slopes = frame["success_slope"].to_numpy(float)
            mean = float(slopes.mean())
            samples = rng.choice(slopes, size=(10000, len(slopes)), replace=True).mean(axis=1)
            low, high = np.quantile(samples, [.025, .975])
            ax.fill_between(xline, low * xline, high * xline,
                            color=palette[group], alpha=.14)
            ax.plot(xline, mean * xline, color=palette[group], lw=2.5,
                    label=f"{group} mean slope ({mean:.2f})")
            summary_rows.append({"phase": label, "AnalysisGroup": group,
                                 "animals": len(slopes), "mean_success_slope": mean,
                                 "bootstrap_ci_lower": low, "bootstrap_ci_upper": high})
        ax.set(title=label, xlabel="Drinking-attempt number",
               ylabel="Cumulative successful attempts", xlim=(0, max_attempt))
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    pd.DataFrame(summary_rows).to_csv(output / "cumulative_learning_group_summary.csv", index=False)
    fig.suptitle("Cumulative place-learning success by drinking attempt")
    _save(fig, output / "cumulative_learning.png")


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
    # 19:00-07:00 wraps midnight, so it needs both spans; the evening one must
    # run to 24 or the 23:00 bin is silently left outside the shaded dark phase.
    ax.axvspan(19, 24, color="0.94", zorder=0, label="Nominal dark phase")
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
    # Rows must be calendar-continuous: a day with no visits still gets a blank
    # row, otherwise adjacent rows are not adjacent days and the actogram lies
    # about phase. The right half is the FOLLOWING day, so the last row's right
    # half is blank rather than wrapping around to the first day.
    all_dates = pd.date_range(min(act["date"]), max(act["date"]), freq="D").date
    for animal in animals:
        frame = act[act["AnimalName"].eq(animal)]
        matrix = frame.pivot(index="date", columns="hour", values="visits").reindex(
            index=all_dates, columns=range(24), fill_value=0
        ).fillna(0).to_numpy()
        following = np.vstack([matrix[1:], np.zeros((1, 24))])
        matrices[animal] = np.concatenate([matrix, following], axis=1)
    date_labels = [str(d) for d in all_dates]
    common_max = max(float(matrix.max()) for matrix in matrices.values())
    image = None
    for ax, animal in zip(axes[:, 0], animals):
        image = ax.imshow(matrices[animal], aspect="auto", interpolation="nearest",
                          cmap="Greys", vmin=0, vmax=max(1, common_max))
        ax.set(ylabel=animal, xticks=[0, 12, 24, 36, 47], xticklabels=["0", "12", "24", "36", "48"],
               yticks=range(len(date_labels)), yticklabels=date_labels)
        ax.tick_params(axis="y", labelsize=7)
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


def tier2(session: Session, output: Path, block_size: int = 100,
          threshold: float = 0.5, consecutive: int = 2) -> None:
    output.mkdir(parents=True, exist_ok=True)
    daily = daily_learning(session.visits)
    # Each animal-day rests on its own number of choices, so it gets its own
    # boundary: a 45% day on 20 visits is chance, the same value on 300 is not.
    bounds = boundary_frame(daily["conditioned_visits"])
    daily = pd.concat([daily, bounds[["chance_lower", "chance_upper"]]], axis=1)
    daily["above_chance"] = daily["accuracy"].gt(daily["chance_upper"])
    daily["below_chance"] = daily["accuracy"].lt(daily["chance_lower"])
    daily.to_csv(output / "daily_learning.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for animal, frame in daily.groupby("AnimalName"):
        ax.plot(pd.to_datetime(frame["date"]), frame["accuracy"], marker="o", alpha=.55, lw=1, label=animal)
    marked = daily[daily["above_chance"].fillna(False)]
    if not marked.empty:
        ax.scatter(pd.to_datetime(marked["date"]), marked["accuracy"], s=90, facecolors="none",
                   edgecolors="black", linewidths=1.2, zorder=6, label="Above own boundary")
    cohort = daily.groupby("date").agg(mean=("accuracy", "mean"), sem=("accuracy", "sem"),
                                       n=("conditioned_visits", "median")).reset_index()
    dates = pd.to_datetime(cohort["date"])
    envelope = boundary_frame(cohort["n"])
    ax.fill_between(dates, envelope["chance_lower"].fillna(0), envelope["chance_upper"],
                    color="0.75", alpha=.35, zorder=0,
                    label="Chance range (binomial, p=0.25, α=0.05)")
    ax.plot(dates, envelope["chance_upper"], color="0.45", ls="--", lw=1, zorder=1)
    ax.plot(dates, cohort["mean"], color="black", lw=2.5, label="Cohort mean", zorder=4)
    ax.fill_between(dates, cohort["mean"]-cohort["sem"], cohort["mean"]+cohort["sem"],
                    color="black", alpha=.15, zorder=3)
    ax.set(xlabel="Date", ylabel="Correct-place visits / conditioned visits", ylim=(0, 1),
           title="Place-learning acquisition")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    _save(fig, output / "daily_learning.png")
    blocks = visit_block_learning(session.visits, block_size)
    block_bounds = boundary_frame(blocks["visits"])
    blocks = pd.concat([blocks, block_bounds[["chance_lower", "chance_upper"]]], axis=1)
    blocks["above_chance"] = blocks["accuracy"].gt(blocks["chance_upper"])
    blocks.to_csv(output / "visit_block_learning.csv", index=False)
    lower, upper = chance_boundary(block_size)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhspan(0 if np.isnan(lower) else lower, upper, color="0.75", alpha=.35, zorder=0,
               label=f"Chance range, n={block_size}")
    ax.axhline(upper, ls="--", color="0.45", lw=1, zorder=1)
    ax.axhline(CHANCE, ls=":", color="0.55", lw=1, zorder=1, label="p = 0.25")
    for animal, frame in blocks.groupby("AnimalName"):
        ax.plot(frame["block"] * block_size, frame["accuracy"], marker="o", ms=3, label=animal, zorder=3)
    ax.set(xlabel="Conditioned visits completed", ylabel="Correct-place proportion", ylim=(0, 1),
           title=f"Visit-block acquisition (above {upper:.2f} is above chance)")
    ax.legend(frameon=False, fontsize=7, ncol=2); _save(fig, output / "visit_block_learning.png")
    criterion = trials_to_criterion(blocks, threshold=threshold, consecutive=consecutive, block_size=block_size)
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
    ax.set(xticks=positions, xticklabels=criterion["AnimalName"], ylabel="Conditioned visits to criterion",
           xlim=(-.5, len(criterion) - .5),
           title=f"Criterion: ≥{threshold:.0%} for {consecutive} complete {block_size}-visit blocks")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output / "trials_to_criterion.png")
    # Terminal accuracy must come from a COMPLETE block. The trailing block of a
    # session is whatever visits were left over -- 3 or 8 visits is common -- and
    # plotting that beside a full block invites reading binomial noise as the
    # best-performing mouse. Animals without a complete block are reported as
    # such instead of being given a number.
    complete = blocks[blocks["visits"].ge(block_size)]
    terminal = complete.groupby("AnimalName").tail(1)
    dropped = sorted(set(blocks["AnimalName"]) - set(terminal["AnimalName"]))
    if dropped:
        absent = blocks.groupby("AnimalName").tail(1)
        absent = absent[absent["AnimalName"].isin(dropped)].copy()
        absent["accuracy"] = np.nan
        terminal = pd.concat([terminal, absent]).sort_values("AnimalName")
    terminal = terminal.rename(columns={"visits": "block_visits"})
    terminal["complete_block"] = terminal["block_visits"].ge(block_size)
    terminal_bounds = boundary_frame(terminal["block_visits"])
    terminal["chance_lower"] = terminal_bounds["chance_lower"].to_numpy()
    terminal["chance_upper"] = terminal_bounds["chance_upper"].to_numpy()
    terminal["above_chance"] = terminal["accuracy"].gt(terminal["chance_upper"])
    terminal.to_csv(output / "terminal_accuracy.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    positions = np.arange(len(terminal))
    ax.scatter(positions, terminal["accuracy"], c=[f"C{i}" for i in positions], s=60)
    for position, (_, row) in zip(positions, terminal.iterrows()):
        if row["complete_block"]:
            ax.annotate(f"n={int(row['block_visits'])}", (position, row["accuracy"]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color="0.35")
        else:
            ax.text(position, .04, f"no complete block (n={int(row['block_visits'])})", rotation=90,
                    ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=7, color="0.4")
    ax.axhspan(0 if np.isnan(lower) else lower, upper, color="0.75", alpha=.35, zorder=0,
               label=f"Chance range, n={block_size}")
    ax.axhline(upper, ls="--", color="0.45", lw=1, zorder=1)
    ax.axhline(CHANCE, ls=":", color="0.55", lw=1, zorder=1)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set(xticks=positions, xticklabels=terminal["AnimalName"], ylabel="Correct-place proportion",
           xlim=(-.5, len(terminal) - .5), ylim=(0, 1),
           title=f"Terminal complete {block_size}-visit block (above chance: >{upper:.2f})")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, output / "terminal_accuracy.png")
    x = add_time_fields(session.visits)
    # PlaceError is 0 on unconditioned visits too, so its raw mean understates the
    # error rate whenever a session mixes neutral and conditioned visits. Score it
    # on the same denominator as accuracy.
    conditioned = x[x["conditioned"]]
    errors = conditioned.groupby(["AnimalName", "date"]).agg(
        place_error_rate=("PlaceError", "mean"), visits=("VisitID", "size"),
        accuracy=("correct", "mean")).reset_index()
    errors = errors.merge(x.groupby(["AnimalName", "date"]).size().rename("all_visits").reset_index(),
                          on=["AnimalName", "date"], how="left")
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
        ax.scatter(frame["all_visits"], frame["accuracy"], label=animal)
    ax.axhline(CHANCE, ls=":", color="0.55", label="p = 0.25 (n varies per point; see daily_learning.csv for per-point boundaries)")
    ax.set(xlabel="Visits per day", ylabel="Correct-place proportion", ylim=(0, 1), title="Activity–accuracy dissociation")
    ax.legend(frameon=False)
    _save(fig, output / "activity_accuracy.png")
