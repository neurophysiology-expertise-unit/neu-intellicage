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
    ax.text(.99, .97, "A = acquisition target; O = opposite target", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="0.35")
    ax.axhline(.25, ls="--", color="0.5"); ax.set(xlabel="Date", ylabel="Visit proportion", ylim=(0, 1),
        title="Performance under the recorded alternating target schedule")
    ax.legend(frameon=False, ncol=2); fig.tight_layout(); fig.savefig(output / "programmed_target_performance.png", dpi=180); plt.close(fig)


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
                   f"![Hourly activity](sessions/{sid}/tier1/hourly_activity.png)", ""]
        if next(x for x in summaries if x["session"] == sid)["conditioned_visits"]:
            report += [f"![Daily place accuracy](sessions/{sid}/tier2/daily_learning.png)", "",
                       f"![Programmed target corner](sessions/{sid}/target_corner_by_day.png)", "",
                       f"![Performance under recorded schedule](sessions/{sid}/programmed_target_performance.png)", ""]
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
