from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import load_session
from .metrics import add_time_fields, daily_learning
from .plots import qc, tier1, tier2
from .provenance import write_provenance


def _exclude_animals(session, names: list[str]):
    """Return a session with named non-experimental animals removed."""
    if not names:
        return session
    keep_visits = session.visits.loc[~session.visits["AnimalName"].isin(names)].copy()
    keep_ids = set(keep_visits["VisitID"])
    keep_animals = session.animals.loc[~session.animals["AnimalName"].isin(names)].copy()
    keep_nosepokes = session.nosepokes.loc[session.nosepokes["VisitID"].isin(keep_ids)].copy() if not session.nosepokes.empty else session.nosepokes
    return replace(session, animals=keep_animals, visits=keep_visits, nosepokes=keep_nosepokes)


def _target_by_day(visits: pd.DataFrame) -> pd.DataFrame:
    x = add_time_fields(visits)
    correct = x[x["CornerCondition"].eq(1)]
    if correct.empty:
        return pd.DataFrame(columns=["AnimalName", "date", "target_corner", "target_visits"])
    counts = correct.groupby(["AnimalName", "date", "Corner"]).size().rename("target_visits").reset_index()
    maximum = counts.groupby(["AnimalName", "date"])["target_visits"].transform("max")
    return counts[counts["target_visits"].eq(maximum)].rename(columns={"Corner": "target_corner"}).reset_index(drop=True)


def _session_overview(session, output: Path) -> dict:
    x = add_time_fields(session.visits)
    daily = daily_learning(session.visits)
    nose = session.nosepokes.copy()
    if not nose.empty:
        nose = nose.merge(session.visits[["VisitID", "AnimalName"]], on="VisitID", how="left")
    summary = {
        "animals": int(x["AnimalName"].nunique()), "visits": int(len(x)),
        "nosepokes": int(len(nose)), "start": x["Start"].min().isoformat(),
        "end": x["End"].max().isoformat(), "conditioned_visits": int(x["CornerCondition"].ne(0).sum()),
    }
    rows = x.groupby("AnimalName").agg(visits=("VisitID", "size"),
        first_visit=("Start", "min"), last_visit=("End", "max")).reset_index()
    if not nose.empty:
        np_counts = nose.groupby("AnimalName").size().rename("nosepokes")
        rows = rows.merge(np_counts, on="AnimalName", how="left")
    rows["nosepokes"] = rows.get("nosepokes", 0)
    rows.to_csv(output / "animal_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].bar(rows["AnimalName"], rows["visits"])
    axes[0, 0].set(title="Visits by animal", ylabel="Visits")
    axes[0, 1].bar(rows["AnimalName"], rows["nosepokes"])
    axes[0, 1].set(title="Nosepokes by animal", ylabel="Nosepokes")
    corners = x.groupby(["AnimalName", "Corner"]).size().unstack(fill_value=0).reindex(columns=range(1, 5), fill_value=0)
    bottom = np.zeros(len(corners))
    for corner in corners.columns:
        axes[1, 0].bar(corners.index, corners[corner], bottom=bottom, label=f"Corner {corner}")
        bottom += corners[corner].to_numpy()
    axes[1, 0].set(title="Corner use", ylabel="Visits"); axes[1, 0].legend(frameon=False, fontsize=8)
    for animal, frame in daily.groupby("AnimalName"):
        axes[1, 1].plot(pd.to_datetime(frame["date"]), frame["visits"], marker="o", label=animal)
    axes[1, 1].set(title="Daily activity", ylabel="Visits", xlabel="Date")
    for ax in axes.flat: ax.tick_params(axis="x", rotation=25)
    fig.suptitle(session.path.name); fig.tight_layout(); fig.savefig(output / "session_overview.png", dpi=180); plt.close(fig)
    return summary


def _programmed_target_analysis(session, output: Path) -> None:
    """Show performance against the target actually recorded on each day."""
    x = add_time_fields(session.visits)
    targets = _target_by_day(session.visits)
    first = targets.sort_values("date").groupby("AnimalName").first()["target_corner"]
    x["acquisition_corner"] = x["AnimalName"].map(first)
    x["opposite_corner"] = ((x["acquisition_corner"] + 1) % 4) + 1
    target_map = targets.set_index(["AnimalName", "date"])["target_corner"]
    x["programmed_corner"] = [target_map.get((a, d), pd.NA) for a, d in zip(x["AnimalName"], x["date"])]
    x["target_state"] = np.where(x["programmed_corner"].eq(x["acquisition_corner"]), "acquisition target",
                          np.where(x["programmed_corner"].eq(x["opposite_corner"]), "opposite target", "ambiguous"))
    x["at_acquisition_corner"] = x["Corner"].eq(x["acquisition_corner"])
    x["at_opposite_corner"] = x["Corner"].eq(x["opposite_corner"])
    daily = x.groupby(["AnimalName", "date", "target_state"]).agg(
        visits=("VisitID", "size"), current_target_accuracy=("correct", "mean"),
        acquisition_corner_preference=("at_acquisition_corner", "mean"),
        opposite_corner_preference=("at_opposite_corner", "mean"),
    ).reset_index()
    daily.to_csv(output / "programmed_target_performance.csv", index=False)
    corner_daily = x.groupby(["AnimalName", "date", "Corner"]).size().rename("corner_visits").reset_index()
    corner_daily["total_visits"] = corner_daily.groupby(["AnimalName", "date"])["corner_visits"].transform("sum")
    corner_daily["preference"] = corner_daily["corner_visits"] / corner_daily["total_visits"]
    corner_daily = corner_daily.merge(targets[["AnimalName", "date", "target_corner"]],
                                      on=["AnimalName", "date"], how="left")
    corner_daily["is_programmed_target"] = corner_daily["Corner"].eq(corner_daily["target_corner"])
    corner_daily.to_csv(output / "individual_daily_corner_preference.csv", index=False)

    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    animals = list(corner_daily["AnimalName"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, animal in zip(axes.flat, animals):
        frame = corner_daily[corner_daily["AnimalName"].eq(animal)]
        for corner in range(1, 5):
            part = frame[frame["Corner"].eq(corner)]
            ax.plot(pd.to_datetime(part["date"]), part["preference"], marker="o",
                    color=colors[corner], label=f"C{corner}")
        target = frame[frame["is_programmed_target"]]
        ax.scatter(pd.to_datetime(target["date"]), target["preference"], s=105,
                   facecolors="none", edgecolors="black", linewidths=1.5,
                   zorder=5, label="Programmed target")
        ax.axhline(.25, ls="--", color="0.55", lw=1)
        ax.set(title=animal, ylim=(0, 1), ylabel="Daily visit proportion")
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("Daily corner preference by individual animal\nBlack ring = target recorded by controller")
    fig.tight_layout(); fig.savefig(output / "individual_daily_corner_preference.png", dpi=180); plt.close(fig)

    for animal in animals:
        frame = corner_daily[corner_daily["AnimalName"].eq(animal)].copy()
        dates_for_animal = sorted(frame["date"].unique())
        xpos = np.arange(len(dates_for_animal)); width = .19
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for corner in range(1, 5):
            part = frame[frame["Corner"].eq(corner)].set_index("date").reindex(dates_for_animal)
            bars = ax.bar(xpos + (corner - 2.5) * width, part["preference"], width,
                          color=colors[corner], label=f"Corner {corner}")
            for bar, is_target in zip(bars, part["is_programmed_target"].fillna(False)):
                if is_target:
                    bar.set_edgecolor("black"); bar.set_linewidth(2.2)
        ax.axhline(.25, ls="--", color="0.45", lw=1)
        ax.set(xticks=xpos, xticklabels=[str(d)[5:] for d in dates_for_animal], ylim=(0, 1),
               xlabel="Date", ylabel="Visit proportion", title=f"{animal}: daily corner preference")
        ax.legend(frameon=False, ncol=4); fig.tight_layout()
        safe_name = animal.lower().replace(" ", "_")
        fig.savefig(output / f"{safe_name}_daily_corner_preference.png", dpi=180); plt.close(fig)

    daily["daily_change_pp"] = daily.groupby("AnimalName")["current_target_accuracy"].diff() * 100
    daily.to_csv(output / "individual_daily_learning_rate.csv", index=False)
    opposite_dates = pd.to_datetime(daily.loc[daily["target_state"].eq("opposite target"), "date"])
    first_opposite = opposite_dates.min() if len(opposite_dates) else pd.Timestamp.max
    acquisition = daily[daily["target_state"].eq("acquisition target") &
                        (pd.to_datetime(daily["date"]) < first_opposite)].copy()
    slopes = []
    for animal, frame in acquisition.groupby("AnimalName"):
        frame = frame.sort_values("date")
        slope = np.polyfit(np.arange(len(frame)), frame["current_target_accuracy"], 1)[0] * 100 if len(frame) > 1 else np.nan
        slopes.append({"AnimalName": animal, "acquisition_slope_pp_per_day": slope,
                       "days": len(frame), "definition": "OLS slope before first opposite-target day; correct-place visit proportion"})
    pd.DataFrame(slopes).to_csv(output / "individual_acquisition_slopes.csv", index=False)
    cohort = daily.groupby(["date", "target_state"]).agg(
        current_target_accuracy=("current_target_accuracy", "mean"),
        acquisition_corner_preference=("acquisition_corner_preference", "mean"),
        opposite_corner_preference=("opposite_corner_preference", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dates = pd.to_datetime(cohort["date"])
    ax.plot(dates, cohort["acquisition_corner_preference"], marker="o", label="Original acquisition corner")
    ax.plot(dates, cohort["opposite_corner_preference"], marker="o", label="Diagonally opposite corner")
    ax.plot(dates, cohort["current_target_accuracy"], color="black", marker="o", lw=2.5, label="Currently programmed target")
    for date, state in zip(dates, cohort["target_state"]):
        ax.text(date, .97, "A" if state == "acquisition target" else "O", ha="center", va="top", fontsize=8)
    ax.axhline(.25, ls="--", color="0.5"); ax.set(xlabel="Date", ylabel="Visit proportion", ylim=(0, 1),
        title="Performance under the recorded alternating target schedule")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.18))
    fig.tight_layout(); fig.savefig(output / "programmed_target_performance.png", dpi=180,
                                    bbox_inches="tight"); plt.close(fig)


def build_experiment_report(config_path: str | Path, output: str | Path) -> Path:
    config = json.loads(Path(config_path).read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = [f"# {config['title']}", "", config.get("scope_note", ""), "",
              "## Session inventory", "",
              "Stage labels below are provisional metadata supplied in the experiment configuration. They are not inferred treatment effects.", "",
              "| Session | Provisional stage | Animals | Visits | Nosepokes | Conditioned visits |", "|---|---|---:|---:|---:|---:|"]
    summaries = []
    for item in config["sessions"]:
        session = load_session(item["path"])
        excluded = item.get("exclude_animals", [])
        session = _exclude_animals(session, excluded)
        session_output = output / "sessions" / item["id"]
        session_output.mkdir(parents=True, exist_ok=True)
        summary = _session_overview(session, session_output)
        summary.update({"session": item["id"], "stage": item["stage"], "source": str(session.path)})
        summaries.append(summary)
        qc(session, session_output / "qc")
        tier1(session, session_output / "tier1")
        if summary["conditioned_visits"]:
            tier2(session, session_output / "tier2", config.get("block_size", 100))
            targets = _target_by_day(session.visits)
            targets.to_csv(session_output / "target_corner_by_day.csv", index=False)
            pivot = targets.pivot(index="date", columns="AnimalName", values="target_corner")
            fig, ax = plt.subplots(figsize=(9, 4))
            for animal in pivot:
                ax.plot(pd.to_datetime(pivot.index), pivot[animal], marker="o", label=animal)
            ax.set(yticks=[1, 2, 3, 4], ylabel="Programmed correct corner", xlabel="Date", title="Target corner recorded in visit conditions")
            ax.legend(frameon=False); fig.tight_layout(); fig.savefig(session_output / "target_corner_by_day.png", dpi=180); plt.close(fig)
            _programmed_target_analysis(session, session_output)
        write_provenance(session_output, session.path, {"stage": item["stage"], "block_size": config.get("block_size", 100), "excluded_animals": excluded})
        report.append(f"| {item['id']} | {item['stage']} | {summary['animals']} | {summary['visits']:,} | {summary['nosepokes']:,} | {summary['conditioned_visits']:,} |")
    pd.DataFrame(summaries).to_csv(output / "session_summary.csv", index=False)
    report += ["", "## Quick-look figures", ""]
    for item in config["sessions"]:
        sid = item["id"]
        report += [f"### {sid} — {item['stage']}", "", item.get("note", ""), "",
                   f"![Session overview](sessions/{sid}/session_overview.png)", "",
                   "*Session overview.* Four quality-control views: total visits and nosepokes per animal, the distribution of visits across the four physical corners, and visits per calendar day. Use it to identify unusually inactive animals, corner bias, incomplete days, or gross recording problems; it does not by itself measure learning.", "",
                   f"![Visits per animal and day](sessions/{sid}/qc/visits_per_animal_day.png)", "",
                   "*Visits per animal per day.* Each line is one mouse and each point is its number of corner visits that calendar day. This is primarily a recording/activity check. The first and last dates are partial recording days and should not be compared directly with complete days.", "",
                   f"![Hourly activity](sessions/{sid}/tier1/hourly_activity.png)", "",
                   "*Hourly activity profile.* For each mouse, visits are converted to visits per observed hour and then averaged by clock hour. The line is the animal mean and the blue band is SEM across animals. Gray indicates the nominal dark phase (19:00–07:00), which must be checked against the actual room light schedule before circadian interpretation.", "",
                   f"![Double-plotted actograms](sessions/{sid}/tier1/actograms.png)", "",
                   "*Double-plotted actograms.* Each row is a recording day shown across 48 hours by repeating the following day alongside it. Darker pixels mean more visits. This makes daily timing, phase shifts and disrupted activity patterns visible; partial boundary days and unverified lighting limit formal interpretation.", "",
                   f"![Inter-visit intervals](sessions/{sid}/tier1/inter_visit_intervals.png)", "",
                   "*Inter-visit interval distribution.* Time between consecutive visits for each animal, shown on a logarithmic x-axis. Leftward distributions indicate rapid repeated visiting; long right tails indicate extended inactive periods. This is an activity-organization measure, not a memory score.", ""]
        if next(x for x in summaries if x["session"] == sid)["conditioned_visits"]:
            report += [f"![Daily place accuracy](sessions/{sid}/tier2/daily_learning.png)", "",
                       "*Daily programmed-corner accuracy.* Thin colored lines are individual mice; the black line and band are the cohort mean ± SEM. The value is the proportion of visits to the corner marked correct by the controller on that day, with 25% as the four-corner reference. Because the controller alternated targets after 16 July, this is performance under the executed schedule—not a continuous reversal-learning curve.", "",
                       f"![Programmed target corner](sessions/{sid}/target_corner_by_day.png)", "",
                       "*Programmed target audit.* The correct corner reconstructed from `CornerCondition` for every mouse and day. This is a configuration/QC plot rather than an animal-performance plot. It reveals the unintended original–opposite alternation on 17–20 July.", "",
                       f"![Individual daily corner preference](sessions/{sid}/individual_daily_corner_preference.png)", "",
                       "*Individual daily corner preference.* One panel per mouse; each colored curve is the proportion of that day's visits made to one physical corner. Black rings mark the corner programmed as correct. This is the closest reconstruction of the IntelliCage preference display while retaining day-by-day resolution. Separate bar-chart files for each animal are stored beside this figure.", "",
                       f"![Performance under recorded schedule](sessions/{sid}/programmed_target_performance.png)", "",
                       "*Original versus opposite target.* Cohort-average preference for the original acquisition corner (blue), its diagonal opposite (orange), and the currently programmed target (black). A/O labels show which contingency ran that day. It demonstrates that behavior redirected on opposite-target days but cannot substitute for a sustained reversal experiment.", "",
                       f"![Visit-block learning](sessions/{sid}/tier2/visit_block_learning.png)", "",
                       "*Visit-block learning.* Correct-place proportion in consecutive blocks of 100 visits per mouse. This gives finer acquisition resolution than calendar days and avoids treating a quiet day as equally informative as a busy day. Blocks spanning a midnight target change require caution; they are not phase-pure.", "",
                       f"![Activity and accuracy](sessions/{sid}/tier2/activity_accuracy.png)", "",
                       "*Activity–accuracy relationship.* Each point is one mouse-day: total visits on the x-axis and currently programmed-corner accuracy on the y-axis. It helps distinguish altered activity from altered choice accuracy, but with four control animals it is descriptive and is not a treatment-effect test.", "",
                       f"![Error decomposition](sessions/{sid}/tier2/error_decomposition.png)", "",
                       "*Error decomposition.* Daily cohort mean place-error rate and side-error rate. Place errors use visits as the denominator; side errors use nosepokes, so their absolute magnitudes are not directly interchangeable. The plot identifies when spatial versus local left/right responding contributes to errors.", "",
                       f"![Terminal accuracy](sessions/{sid}/tier2/terminal_accuracy.png)", "",
                       "*Terminal accuracy.* One dot per mouse for its final, potentially incomplete, 100-visit block; the dashed line is 25%. Because the experiment returned to the acquisition target on 20 July, this is terminal performance under that restored target—not terminal reversal accuracy.", "",
                       f"![Trials to criterion](sessions/{sid}/tier2/trials_to_criterion.png)", "",
                       "*Trials to criterion.* Visits required to produce two consecutive 100-visit blocks at ≥50% correct. Missing dots mean the operational criterion was not reached. The threshold is a transparent descriptive rule, not a literature-derived inferential test, and target changes can reset its biological meaning.", ""]
    report += ["## Protocol diagnosis", "",
               "The intended protocol was ordinary place reversal: maintain one correct corner during acquisition, switch once to the diagonally opposite corner, and keep that new corner active while the animal relearns. This is also how Voikar et al. (2018) describe Place/Reversal and how Roos et al. (2026) distinguish place-preference reversal from serial reversal.", "",
               "The raw visit conditions instead encode this daily sequence: acquisition target on 13–16 July, opposite target on 17 July, acquisition target on 18 July, opposite target on 19 July, and acquisition target on 20 July. Because the archived experiment notes explicitly say the opposite target should hold, this is treated as a programming error in the day-pattern chain—not as the intended reversal procedure.", "",
               "Consequently, 13–16 July can be used as acquisition. The isolated opposite-target days on 17 and 19 July and return days on 18 and 20 July are reported descriptively as alternating contingencies; they cannot establish a sustained reversal-learning curve.", ""]
    report += ["## Interpretation status", "",
               "This is a descriptive screening report. Treatment identity, light/dark schedule, exact phase-switch timestamps, reward contingencies, and whether the July animals belong to the Tau study require confirmation before inferential analysis.", "",
               "The programmed correct corner changes within the place-learning export. Therefore a single session-wide learning slope is not interpreted as acquisition or reversal until the intended schedule is confirmed.", ""]
    report += ["## Analysis provenance", "",
               "- Session inventory, visit counts, nosepoke counts, corner use and hardware-event summaries are direct quality-control views of the exported tables.",
               "- Individual/cohort learning curves, visit blocks, terminal accuracy, trials-to-criterion, error decomposition and activity–accuracy separation implement the outcomes pre-specified in `projects/intellicage/plan.md` §8.",
               "- Actograms and IS/IV/RA circadian summaries were motivated by the spontaneous-activity signature discussed from Voikar et al. (2018) in that plan; they are descriptive here because the lighting schedule has not been verified.",
               "- Phase names, the intended 17 July 00:01 reversal, four balanced target corners and the seven-second nosepoke door rule came from the archived learning log/pilot report and NewBehavior place-learning tutorial.",
               "- No values were copied from the earlier AI reports. Plotted values were recomputed from `Visits.txt`, `Nosepokes.txt`, `Animals.txt`, and `HardwareEvents.txt`. Where the earlier narrative conflicts with those tables, the raw export controls the current report.", ""]
    path = output / "report.md"; path.write_text("\n".join(report))
    return path
