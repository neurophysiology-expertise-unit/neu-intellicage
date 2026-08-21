"""Focused, explicitly requested follow-up analyses for an experiment report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import Session
from .groups import compare_many
from .metrics import add_time_fields


def _clock_hour(timestamp: pd.Series) -> pd.Series:
    return timestamp.dt.hour + timestamp.dt.minute / 60 + timestamp.dt.second / 3600


def all_sessions_light_dark_activity(
    sessions: dict[str, Session], output: Path, groups: dict[str, list[str]],
    illumination_threshold: float = 10.0,
) -> dict:
    """Combine activity across sessions without counting inter-session gaps."""
    output.mkdir(parents=True, exist_ok=True)
    animals = [animal for members in groups.values() for animal in members]
    animal_group = {animal: group for group, members in groups.items() for animal in members}
    daily_parts = []
    for session_id, session in sessions.items():
        visits = session.visits.sort_values("Start").copy()
        environment = pd.read_csv(session.path / "IntelliCage" / "Environment.txt", sep="\t",
                                  usecols=["DateTime", "Illumination"])
        environment["DateTime"] = pd.to_datetime(environment["DateTime"], errors="raise")
        environment = environment.sort_values("DateTime")
        environment["lights_on"] = environment["Illumination"].gt(illumination_threshold)
        start, end = visits["Start"].min(), visits["End"].max()
        environment["next_time"] = environment["DateTime"].shift(-1).fillna(end)
        environment["interval_start"] = environment["DateTime"].clip(lower=start)
        environment["interval_end"] = environment["next_time"].clip(upper=end)
        environment["duration_h"] = (environment["interval_end"] - environment["interval_start"]).dt.total_seconds().clip(lower=0) / 3600
        environment["date"] = environment["interval_start"].dt.date
        exposure = environment.groupby(["date", "lights_on"])["duration_h"].sum().unstack(fill_value=0)
        exposure = exposure.reindex(columns=[False, True], fill_value=0)

        tagged = pd.merge_asof(visits, environment[["DateTime", "lights_on"]],
                               left_on="Start", right_on="DateTime", direction="backward")
        tagged["date"] = tagged["Start"].dt.date
        counts = tagged.groupby(["AnimalName", "date", "lights_on"]).size().unstack(fill_value=0)
        index = pd.MultiIndex.from_product([animals, exposure.index], names=["AnimalName", "date"])
        counts = counts.reindex(index=index, columns=[False, True], fill_value=0)
        frame = counts.rename(columns={False: "visits_dark", True: "visits_light"}).reset_index()
        frame["dark_hours"] = frame["date"].map(exposure[False])
        frame["light_hours"] = frame["date"].map(exposure[True])
        frame["visits_per_dark_hour"] = frame["visits_dark"] / frame["dark_hours"].replace(0, np.nan)
        frame["visits_per_light_hour"] = frame["visits_light"] / frame["light_hours"].replace(0, np.nan)
        frame["light_dark_rate_ratio"] = frame["visits_per_light_hour"] / frame["visits_per_dark_hour"].replace(0, np.nan)
        frame["daily_ratio_has_sufficient_exposure"] = frame["dark_hours"].ge(3) & frame["light_hours"].ge(3)
        frame["session"] = session_id
        frame["Group"] = frame["AnimalName"].map(animal_group)
        daily_parts.append(frame)
    daily = pd.concat(daily_parts, ignore_index=True)
    daily.to_csv(output / "all_august_daily_light_dark_activity.csv", index=False)

    combined = daily.groupby(["AnimalName", "Group"], as_index=False).agg(
        visits_dark=("visits_dark", "sum"), visits_light=("visits_light", "sum"),
        dark_hours=("dark_hours", "sum"), light_hours=("light_hours", "sum"))
    combined["visits_per_dark_hour"] = combined["visits_dark"] / combined["dark_hours"]
    combined["visits_per_light_hour"] = combined["visits_light"] / combined["light_hours"]
    combined["light_dark_rate_ratio"] = combined["visits_per_light_hour"] / combined["visits_per_dark_hour"]
    combined.to_csv(output / "all_august_light_dark_activity_by_animal.csv", index=False)

    palette = {name: f"C{i}" for i, name in enumerate(groups)}
    fig, (daily_ax, total_ax) = plt.subplots(1, 2, figsize=(12, 4.8))
    for animal, frame in daily.groupby("AnimalName"):
        group = animal_group[animal]
        frame = frame[frame["daily_ratio_has_sufficient_exposure"]].sort_values("date")
        daily_ax.plot(pd.to_datetime(frame["date"]), frame["light_dark_rate_ratio"], marker="o",
                      color=palette[group], alpha=.7, lw=1.3)
        last = frame.dropna(subset=["light_dark_rate_ratio"]).tail(1)
        if not last.empty:
            daily_ax.text(pd.to_datetime(last["date"].iloc[0]), last["light_dark_rate_ratio"].iloc[0],
                          animal.replace("Animal ", "A"), color=palette[group], fontsize=8)
    daily_ax.axvline(pd.Timestamp("2026-08-18"), color="0.35", ls="--", lw=1,
                     label="Opposite target begins")
    daily_ax.set(ylabel="Daily light:dark visit-rate ratio", xlabel="Date",
                 title="Day-by-day activity organization")
    daily_ax.tick_params(axis="x", rotation=30)
    daily_ax.legend(frameon=False, fontsize=8)

    for position, (group, frame) in enumerate(combined.groupby("Group", sort=False)):
        frame = frame.sort_values("AnimalName")
        offsets = np.linspace(-.12, .12, len(frame))
        total_ax.scatter(position + offsets, frame["light_dark_rate_ratio"],
                         color=palette[group], s=58)
        for x_pos, (_, row) in zip(position + offsets, frame.iterrows()):
            total_ax.annotate(row["AnimalName"].replace("Animal ", "A"),
                              (x_pos, row["light_dark_rate_ratio"]), xytext=(0, 6),
                              textcoords="offset points", ha="center", fontsize=8,
                              color=palette[group])
        total_ax.hlines(frame["light_dark_rate_ratio"].mean(), position-.22, position+.22,
                        color=palette[group], lw=2.5)
    total_ax.axhline(1, color="0.45", ls="--", lw=1)
    total_ax.set(xticks=range(len(groups)), xticklabels=list(groups),
                 ylabel="Combined light:dark visit-rate ratio",
                 title="All recorded August sessions combined")
    fig.suptitle("Activity from habituation through interrupted reversal")
    _save_light_cycle_figure(fig, output / "all_august_light_dark_activity.png")

    comparisons = compare_many(combined, ["visits_per_light_hour", "light_dark_rate_ratio"], groups)
    comparisons.to_csv(output / "all_august_light_dark_comparison.csv", index=False)
    return {"comparisons": comparisons.to_dict("records"), "sessions": list(sessions)}


def correct_visit_actograms(
    session: Session,
    output: Path,
    groups: dict[str, list[str]],
    illumination_threshold: float = 10.0,
) -> dict[str, float]:
    """Correct-visit actograms and illumination-phase summaries.

    ``PlaceError == 0`` is used only where a corner contingency was active;
    neutral visits are not correct trials. Light state is taken from the
    recorded ``Environment.txt`` illumination channel rather than an assumed
    schedule.
    """
    output.mkdir(parents=True, exist_ok=True)
    environment_path = session.path / "IntelliCage" / "Environment.txt"
    environment = pd.read_csv(environment_path, sep="\t", usecols=["DateTime", "Illumination"])
    environment["DateTime"] = pd.to_datetime(environment["DateTime"], errors="raise")
    environment = environment.sort_values("DateTime")
    environment["lights_on"] = environment["Illumination"].gt(illumination_threshold)
    environment["transition"] = environment["lights_on"].ne(environment["lights_on"].shift())
    transitions = environment.loc[environment["transition"], ["DateTime", "Illumination", "lights_on"]].copy()
    transitions["clock_hour"] = _clock_hour(transitions["DateTime"])
    transitions.to_csv(output / "recorded_light_transitions.csv", index=False)

    visits = add_time_fields(session.visits).sort_values("Start")
    correct = visits[visits["conditioned"] & visits["correct"].fillna(False)].copy()
    correct = pd.merge_asof(
        correct.sort_values("Start"),
        environment[["DateTime", "Illumination", "lights_on"]],
        left_on="Start", right_on="DateTime", direction="backward",
    )
    correct["date"] = correct["Start"].dt.date
    correct["hour"] = correct["Start"].dt.hour

    animals = [animal for members in groups.values() for animal in members
               if animal in set(visits["AnimalName"])]
    all_dates = pd.date_range(visits["Start"].min().normalize(),
                              visits["Start"].max().normalize(), freq="D").date
    full_index = pd.MultiIndex.from_product(
        [animals, all_dates, range(24)], names=["AnimalName", "date", "hour"]
    )
    hourly = (correct.groupby(["AnimalName", "date", "hour"]).size().rename("correct_visits")
              .reindex(full_index, fill_value=0).reset_index())
    hourly.to_csv(output / "correct_visits_by_animal_day_hour.csv", index=False)

    # Exposure duration in each recorded lighting state. Environment is sampled
    # every minute; interval weighting retains the actual transition times.
    end = visits["End"].max()
    environment["next_time"] = environment["DateTime"].shift(-1).fillna(end)
    environment["duration_h"] = (
        environment["next_time"].clip(upper=end) - environment["DateTime"]
    ).dt.total_seconds().clip(lower=0) / 3600
    exposure = environment.groupby("lights_on")["duration_h"].sum()
    light_h = float(exposure.get(True, np.nan))
    dark_h = float(exposure.get(False, np.nan))

    phase = correct.groupby(["AnimalName", "lights_on"]).size().unstack(fill_value=0)
    phase = phase.reindex(index=animals, columns=[False, True], fill_value=0)
    summary = pd.DataFrame({
        "AnimalName": animals,
        "Group": [next(name for name, members in groups.items() if animal in members)
                  for animal in animals],
        "correct_visits_dark": phase[False].to_numpy(),
        "correct_visits_light": phase[True].to_numpy(),
    })
    all_with_light = pd.merge_asof(
        visits.sort_values("Start"),
        environment[["DateTime", "Illumination", "lights_on"]],
        left_on="Start", right_on="DateTime", direction="backward",
    )
    all_phase = all_with_light.groupby(["AnimalName", "lights_on"]).size().unstack(fill_value=0)
    all_phase = all_phase.reindex(index=animals, columns=[False, True], fill_value=0)
    summary["all_visits_dark"] = all_phase[False].to_numpy()
    summary["all_visits_light"] = all_phase[True].to_numpy()
    summary["all_visits_per_dark_hour"] = summary["all_visits_dark"] / dark_h
    summary["all_visits_per_light_hour"] = summary["all_visits_light"] / light_h
    summary["all_visit_light_dark_rate_ratio"] = (
        summary["all_visits_per_light_hour"] / summary["all_visits_per_dark_hour"]
    )
    summary["correct_visits_total"] = summary["correct_visits_dark"] + summary["correct_visits_light"]
    summary["light_fraction_of_correct_visits"] = (
        summary["correct_visits_light"] / summary["correct_visits_total"]
    )
    summary["correct_visits_per_dark_hour"] = summary["correct_visits_dark"] / dark_h
    summary["correct_visits_per_light_hour"] = summary["correct_visits_light"] / light_h
    summary["light_dark_rate_ratio"] = (
        summary["correct_visits_per_light_hour"] / summary["correct_visits_per_dark_hour"]
    )
    summary["recorded_dark_hours"] = dark_h
    summary["recorded_light_hours"] = light_h
    summary.to_csv(output / "correct_visit_light_phase_by_animal.csv", index=False)

    # Paired rates show the within-mouse day/night change; the ratio panel then
    # makes between-mouse and between-group differences directly comparable.
    palette = {name: f"C{i}" for i, name in enumerate(groups)}
    fig, (rate_ax, ratio_ax) = plt.subplots(1, 2, figsize=(11, 4.8))
    for _, row in summary.iterrows():
        color = palette[row["Group"]]
        rates = [row["all_visits_per_dark_hour"], row["all_visits_per_light_hour"]]
        rate_ax.plot([0, 1], rates, marker="o", color=color, alpha=.72, lw=1.5)
        rate_ax.text(1.035, rates[1], row["AnimalName"].replace("Animal ", "A"),
                     color=color, fontsize=8, va="center")
    for group, color in palette.items():
        rate_ax.plot([], [], marker="o", color=color, label=group)
    rate_ax.set(xticks=[0, 1], xticklabels=["Recorded dark", "Recorded light"],
                xlim=(-.15, 1.22), ylabel="All corner visits per recorded hour",
                title="Paired activity rate within each mouse")
    rate_ax.legend(frameon=False)

    group_names = list(groups)
    offsets = np.linspace(-.12, .12, max(len(summary.groupby("Group").head()), 2))
    for group_position, group in enumerate(group_names):
        frame = summary[summary["Group"].eq(group)].sort_values("AnimalName")
        local_offsets = np.linspace(-.12, .12, len(frame)) if len(frame) > 1 else [0]
        for offset, (_, row) in zip(local_offsets, frame.iterrows()):
            x_pos = group_position + offset
            ratio_ax.scatter(x_pos, row["all_visit_light_dark_rate_ratio"],
                             color=palette[group], s=55, zorder=3)
            ratio_ax.annotate(row["AnimalName"].replace("Animal ", "A"),
                              (x_pos, row["all_visit_light_dark_rate_ratio"]),
                              xytext=(0, 6), textcoords="offset points", ha="center",
                              fontsize=8, color=palette[group])
        ratio_ax.hlines(frame["all_visit_light_dark_rate_ratio"].mean(),
                        group_position - .22, group_position + .22,
                        color=palette[group], lw=2.5)
    ratio_ax.axhline(1, ls="--", color="0.45", lw=1,
                     label="Equal light and dark rates")
    ratio_ax.set(xticks=range(len(group_names)), xticklabels=group_names,
                 ylabel="Light:dark all-visit rate ratio",
                 title="Rest-phase activity relative to dark phase")
    ratio_ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Corner activity under the recorded light–dark cycle")
    _save_light_cycle_figure(fig, output / "all_visit_light_dark_rates.png")

    on_hours = transitions.loc[transitions["lights_on"], "clock_hour"]
    off_hours = transitions.loc[~transitions["lights_on"], "clock_hour"]
    light_on = float(on_hours[on_hours < 12].median())
    light_off = float(off_hours[off_hours > 12].median())

    ncols = len(groups)
    nrows = max(len([a for a in members if a in animals]) for members in groups.values())
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    common_max = max(1.0, float(hourly["correct_visits"].max()))
    image = None
    for col, (group, members) in enumerate(groups.items()):
        present = [a for a in members if a in animals]
        for row in range(nrows):
            ax = axes[row, col]
            if row >= len(present):
                ax.set_visible(False)
                continue
            animal = present[row]
            matrix = (hourly[hourly["AnimalName"].eq(animal)]
                      .pivot(index="date", columns="hour", values="correct_visits")
                      .reindex(index=all_dates, columns=range(24), fill_value=0).to_numpy())
            following = np.vstack([matrix[1:], np.zeros((1, 24))])
            doubled = np.concatenate([matrix, following], axis=1)
            image = ax.imshow(doubled, aspect="auto", interpolation="nearest", cmap="Greys",
                              vmin=0, vmax=common_max)
            for offset in (0, 24):
                ax.axvspan(light_on + offset, light_off + offset,
                           color="#f2c94c", alpha=.13, linewidth=0)
                ax.axvline(light_on + offset, color="#c49a00", lw=.7, alpha=.8)
                ax.axvline(light_off + offset, color="#c49a00", lw=.7, alpha=.8)
            ax.set_title(f"{animal} — {group}", fontsize=10)
            ax.set(xticks=[0, 6, 12, 18, 24, 30, 36, 42, 47],
                   xticklabels=["0", "6", "12", "18", "24", "30", "36", "42", "48"],
                   yticks=range(len(all_dates)), yticklabels=[str(d) for d in all_dates])
            ax.tick_params(axis="y", labelsize=7)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Clock hour (double plotted)")
    fig.suptitle("Correct conditioned visits only\nYellow = recorded lights-on interval")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), pad=.015, fraction=.022)
    colorbar.set_label("Correct visits per hour")
    fig.subplots_adjust(left=.10, right=.90, top=.91, bottom=.08, hspace=.42, wspace=.12)
    fig.savefig(output / "correct_visit_actograms.png", dpi=180)
    plt.close(fig)

    group_summary = summary.groupby("Group").agg(
        animals=("AnimalName", "size"),
        mean_light_correct_rate=("correct_visits_per_light_hour", "mean"),
        sd_light_correct_rate=("correct_visits_per_light_hour", "std"),
        mean_dark_correct_rate=("correct_visits_per_dark_hour", "mean"),
        sd_dark_correct_rate=("correct_visits_per_dark_hour", "std"),
        mean_light_dark_rate_ratio=("light_dark_rate_ratio", "mean"),
        sd_light_dark_rate_ratio=("light_dark_rate_ratio", "std"),
        mean_all_light_rate=("all_visits_per_light_hour", "mean"),
        sd_all_light_rate=("all_visits_per_light_hour", "std"),
        mean_all_dark_rate=("all_visits_per_dark_hour", "mean"),
        sd_all_dark_rate=("all_visits_per_dark_hour", "std"),
        mean_all_light_dark_rate_ratio=("all_visit_light_dark_rate_ratio", "mean"),
        sd_all_light_dark_rate_ratio=("all_visit_light_dark_rate_ratio", "std"),
    ).reset_index()
    group_summary.to_csv(output / "correct_visit_light_phase_by_group.csv", index=False)
    group_sizes = summary.groupby("Group").size()
    comparisons = (compare_many(
        summary,
        ["correct_visits_per_light_hour", "light_dark_rate_ratio",
         "all_visits_per_light_hour", "all_visit_light_dark_rate_ratio"],
        groups,
    ) if len(group_sizes) == 2 and group_sizes.min() >= 2 else pd.DataFrame())
    comparisons.to_csv(output / "correct_visit_light_phase_comparison.csv", index=False)
    return {"light_on": light_on, "light_off": light_off,
            "light_hours": light_h, "dark_hours": dark_h,
            "comparisons": comparisons.to_dict("records")}


def _save_light_cycle_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
