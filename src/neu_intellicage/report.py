from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import load_session
from .metrics import add_time_fields, daily_learning
from .plots import qc, tier1, tier2
from .provenance import write_provenance


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
        write_provenance(session_output, session.path, {"stage": item["stage"], "block_size": config.get("block_size", 100)})
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
                       f"![Programmed target corner](sessions/{sid}/target_corner_by_day.png)", ""]
    report += ["## Interpretation status", "",
               "This is a descriptive screening report. Treatment identity, light/dark schedule, exact phase-switch timestamps, reward contingencies, and whether the July animals belong to the Tau study require confirmation before inferential analysis.", "",
               "The programmed correct corner changes within the place-learning export. Therefore a single session-wide learning slope is not interpreted as acquisition or reversal until the intended schedule is confirmed.", ""]
    path = output / "report.md"; path.write_text("\n".join(report))
    return path
