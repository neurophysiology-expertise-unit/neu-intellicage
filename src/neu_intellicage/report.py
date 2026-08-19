from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import __version__
from .io import load_session
from .metrics import add_time_fields, daily_learning
from .plots import nosepoke_acquisition, qc, tier1, tier2
from .groups import compare_many
from .provenance import write_experiment_provenance, write_provenance


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
    """Reconstruct the programmed target corner per animal and day.

    A day on which the controller switched targets legitimately contains two
    rewarded corners, so the day's target is the corner rewarded on most visits.
    Exactly one row per animal-day is returned: keeping every tied corner
    produced duplicate index entries that crashed the downstream pivot, and a
    switch day with a low-activity mouse is precisely when a tie occurs. Ties are
    broken by the lower corner number and reported in ``ambiguous_target`` so the
    audit can see them rather than being silently given a winner.
    """
    columns = ["AnimalName", "date", "target_corner", "target_visits",
               "rewarded_corners", "ambiguous_target"]
    x = add_time_fields(visits)
    correct = x[x["CornerCondition"].eq(1)]
    if correct.empty:
        return pd.DataFrame(columns=columns)
    counts = correct.groupby(["AnimalName", "date", "Corner"]).size().rename("target_visits").reset_index()
    counts = counts.sort_values(["AnimalName", "date", "target_visits", "Corner"],
                                ascending=[True, True, False, True])
    grouped = counts.groupby(["AnimalName", "date"], sort=False)
    winner = grouped.head(1).rename(columns={"Corner": "target_corner"}).reset_index(drop=True)
    summary = grouped["target_visits"].agg(rewarded_corners="size", top="max",
                                           runner_up=lambda v: sorted(v)[-2] if len(v) > 1 else -1)
    summary = summary.reset_index()
    winner = winner.merge(summary, on=["AnimalName", "date"], how="left")
    winner["ambiguous_target"] = winner["top"].eq(winner["runner_up"])
    return winner[columns]


def _dates_in(frame: pd.DataFrame, dates: list[str] | None) -> pd.DataFrame:
    if not dates:
        return frame
    wanted = {pd.Timestamp(d).date() for d in dates}
    return frame[frame["date"].isin(wanted)]


def _per_animal_measures(session, spec: dict) -> pd.DataFrame:
    """One number per animal for one measure, over an explicit date window.

    Windows are named in the experiment configuration rather than inferred, so
    that "acquisition" or "the opposite-target day" means exactly the dates the
    protocol audit established and nothing shifts silently when data are added.
    """
    kind = spec["kind"]
    x = add_time_fields(session.visits)
    if kind == "nosepoke_probability":
        pokes = session.nosepokes.groupby("VisitID").size().rename("pokes")
        x = x.join(pokes, on="VisitID")
        x["with_poke"] = x["pokes"].fillna(0).gt(0)
        frame = _dates_in(x, spec.get("dates"))
        out = frame.groupby("AnimalName")["with_poke"].mean()
    elif kind == "programmed_target_accuracy":
        frame = _dates_in(x[x["conditioned"]], spec.get("dates"))
        out = frame.groupby("AnimalName")["correct"].mean()
    elif kind == "preference_shift":
        targets = _target_by_day(session.visits)
        first = targets.sort_values("date").groupby("AnimalName").first()["target_corner"]
        x["acquisition_corner"] = x["AnimalName"].map(first)
        x["opposite_corner"] = ((x["acquisition_corner"] + 1) % 4) + 1
        frame = _dates_in(x, spec.get("dates"))
        frame = frame.assign(at_new=frame["Corner"].eq(frame["opposite_corner"]),
                             at_old=frame["Corner"].eq(frame["acquisition_corner"]))
        shift = frame.groupby("AnimalName")[["at_new", "at_old"]].mean()
        out = shift["at_new"] - shift["at_old"]
    elif kind == "daily_accuracy_slope":
        frame = _dates_in(x[x["conditioned"]], spec.get("dates"))
        daily = frame.groupby(["AnimalName", "date"])["correct"].mean().reset_index()
        rows = {}
        for animal, part in daily.groupby("AnimalName"):
            part = part.sort_values("date")
            values = part["correct"].astype(float).to_numpy()
            day = (pd.to_datetime(part["date"]) - pd.to_datetime(part["date"]).min()).dt.days.to_numpy()
            rows[animal] = np.polyfit(day, values, 1)[0] * 100 if len(part) > 1 else np.nan
        out = pd.Series(rows, dtype=float).rename_axis("AnimalName")
    else:
        raise ValueError(f"unknown group measure kind: {kind}")
    return out.rename(spec["name"]).reset_index()


def _group_comparison(config: dict, sessions: dict, output: Path) -> pd.DataFrame:
    """Between-group contrasts for every measure declared in the config."""
    groups = config.get("groups")
    specs = config.get("group_measures", [])
    if not groups or not specs:
        return pd.DataFrame()
    values = None
    for spec in specs:
        measure = _per_animal_measures(sessions[spec["session"]], spec)
        values = measure if values is None else values.merge(measure, on="AnimalName", how="outer")
    values = values.sort_values("AnimalName")
    values.to_csv(output / "group_measures_by_animal.csv", index=False)
    table = compare_many(values, [spec["name"] for spec in specs], groups)
    table.to_csv(output / "group_comparison.csv", index=False)
    return table


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
    # An animal that never poked is absent from the count merge; without this it
    # reads NaN in the CSV and draws no bar at all, which looks like a data gap.
    rows["nosepokes"] = rows["nosepokes"].fillna(0).astype(int) if "nosepokes" in rows else 0
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
    ncols = 2
    nrows = int(np.ceil(len(animals) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
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
    for ax in axes.flat[len(animals):]:
        ax.set_visible(False)
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
    summaries, stamped, loaded = [], [], {}
    for item in config["sessions"]:
        session = load_session(item["path"])
        excluded = item.get("exclude_animals", [])
        session = _exclude_animals(session, excluded)
        loaded[item["id"]] = session
        session_output = output / "sessions" / item["id"]
        session_output.mkdir(parents=True, exist_ok=True)
        summary = _session_overview(session, session_output)
        summary.update({"session": item["id"], "stage": item["stage"], "source": str(session.path)})
        summaries.append(summary)
        qc(session, session_output / "qc")
        tier1(session, session_output / "tier1")
        if summary["nosepokes"]:
            nosepoke_acquisition(session, session_output / "nosepoke")
        if summary["conditioned_visits"] and item.get("run_tier2", True):
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
        stamped.append({"id": item["id"], "source": str(session.path),
                        "provenance": f"sessions/{item['id']}/provenance.json"})
        report.append(f"| {item['id']} | {item['stage']} | {summary['animals']} | {summary['visits']:,} | {summary['nosepokes']:,} | {summary['conditioned_visits']:,} |")
    pd.DataFrame(summaries).to_csv(output / "session_summary.csv", index=False)
    report += ["", "## Quick-look figures", ""]
    for item in config["sessions"]:
        sid = item["id"]
        report += [f"### {sid} — {item['stage']}", "", item.get("note", ""), "",
                   f"![Session overview](sessions/{sid}/session_overview.png)", "",
                   "*Session overview.* Four quality-control views: total visits and nosepokes per animal, the distribution of visits across the four physical corners, and visits per calendar day. Use it to identify unusually inactive animals, corner bias, incomplete days, or gross recording problems; it does not by itself measure learning.", "",
                   f"![Visits per animal and day](sessions/{sid}/qc/visits_per_animal_day.png)", "",
                   "*Visits per animal per day.* Each line is one mouse and each point is its number of corner visits that calendar day. This is primarily a recording/activity check; the accompanying CSV also carries `conditioned_visits` and, where a corner condition was active, accuracy over those visits only. The first and last dates are partial recording days and should not be compared directly with complete days.", "",
                   f"![Hourly activity](sessions/{sid}/tier1/hourly_activity.png)", "",
                   "*Hourly activity profile.* For each mouse, visits are converted to visits per observed hour and then averaged by clock hour. The line is the animal mean and the blue band is SEM across animals. Gray indicates the nominal dark phase (19:00–07:00), which must be checked against the actual room light schedule before circadian interpretation.", "",
                   f"![Double-plotted actograms](sessions/{sid}/tier1/actograms.png)", "",
                   "*Double-plotted actograms.* Each row is a recording day shown across 48 hours by repeating the following day alongside it. The shared grayscale colorbar reports visits per hour, with darker pixels indicating more visits on the same scale for every animal. This makes daily timing, phase shifts and disrupted activity patterns visible; partial boundary days and unverified lighting limit formal interpretation.", "",
                   f"![Inter-visit intervals](sessions/{sid}/tier1/inter_visit_intervals.png)", "",
                   "*Inter-visit interval distribution.* Time between consecutive visits for each animal, shown on a logarithmic x-axis. Leftward distributions indicate rapid repeated visiting; long right tails indicate extended inactive periods. This is an activity-organization measure, not a memory score.", ""]
        if next(x for x in summaries if x["session"] == sid)["nosepokes"]:
            report += [f"![Individual daily nose-poke acquisition](sessions/{sid}/nosepoke/daily_nosepoke_acquisition.png)", "",
                       "*Individual daily nose-poke acquisition.* Each colored line is one mouse. The value is the proportion of its corner visits that contained at least one nose-poke. A mouse whose curve rises earlier began performing the operant response earlier. This is preferable to raw poke totals for acquisition because repeated pokes within one visit cannot dominate the score; `nosepokes_per_visit` is retained in the accompanying CSV as a secondary measure.", ""]
        if (next(x for x in summaries if x["session"] == sid)["conditioned_visits"]
                and item.get("run_tier2", True)):
            report += [f"![Daily place accuracy](sessions/{sid}/tier2/daily_learning.png)", "",
                       item.get("daily_accuracy_caption", "*Daily programmed-corner accuracy.* Thin colored lines are individual mice; the black line and band are the cohort mean ± SEM. The value is the proportion of CONDITIONED visits made to the corner marked correct by the controller on that day, with 25% as the four-corner reference. Visits recorded while no corner condition was active are excluded from the denominator rather than counted as correct. Interpret changes only after checking the controller-recorded target audit below."), "",
                       f"![Programmed target corner](sessions/{sid}/target_corner_by_day.png)", "",
                       item.get("target_audit_caption", "*Programmed target audit.* The correct corner reconstructed from `CornerCondition` for every mouse and day. This is a configuration/QC plot rather than an animal-performance plot; it must agree with the written protocol before learning curves are interpreted."), "",
                       f"![Individual daily corner preference](sessions/{sid}/individual_daily_corner_preference.png)", "",
                       "*Individual daily corner preference.* One panel per mouse; each colored curve is the proportion of that day's visits made to one physical corner. Black rings mark the corner programmed as correct. This is the closest reconstruction of the IntelliCage preference display while retaining day-by-day resolution. Separate bar-chart files for each animal are stored beside this figure.", "",
                       f"![Performance under recorded schedule](sessions/{sid}/programmed_target_performance.png)", "",
                       item.get("programmed_target_caption", "*Original versus opposite target.* Cohort-average preference for the original acquisition corner (blue), its diagonal opposite (orange), and the currently programmed target (black). A/O labels show which contingency was recorded that day. This distinguishes acquisition, opposite-target exposure, and any unintended return to the original target."), "",
                       f"![Visit-block learning](sessions/{sid}/tier2/visit_block_learning.png)", "",
                       "*Visit-block learning.* Correct-place proportion in consecutive blocks of 100 CONDITIONED visits per mouse. This gives finer acquisition resolution than calendar days and avoids treating a quiet day as equally informative as a busy day. Blocks spanning a midnight target change require caution; they are not phase-pure.", "",
                       f"![Activity and accuracy](sessions/{sid}/tier2/activity_accuracy.png)", "",
                       "*Activity–accuracy relationship.* Each point is one mouse-day: total visits on the x-axis and currently programmed-corner accuracy on the y-axis. It helps distinguish altered activity from altered choice accuracy, but with four control animals it is descriptive and is not a treatment-effect test.", "",
                       f"![Error decomposition](sessions/{sid}/tier2/error_decomposition.png)", "",
                       "*Error decomposition.* Daily cohort mean place-error rate and side-error rate. Place errors use visits as the denominator; side errors use nosepokes, so their absolute magnitudes are not directly interchangeable. The plot identifies when spatial versus local left/right responding contributes to errors.", "",
                       f"![Terminal accuracy](sessions/{sid}/tier2/terminal_accuracy.png)", "",
                       item.get("terminal_accuracy_caption", "*Terminal accuracy.* One dot per mouse for its final COMPLETE 100-visit block, annotated with n; the dashed line is 25%. A mouse whose session ended mid-block is marked as having no complete block rather than being scored on the leftover visits. Its biological meaning depends on the target active at the session end and should not be called terminal reversal accuracy if the schedule changed again."), "",
                       f"![Trials to criterion](sessions/{sid}/tier2/trials_to_criterion.png)", "",
                       "*Trials to criterion.* Conditioned visits required to produce two consecutive COMPLETE 100-visit blocks at or above threshold. Partial blocks cannot satisfy the criterion. Missing dots mean the operational criterion was not reached. The threshold is a transparent descriptive rule, not a literature-derived inferential test, and target changes can reset its biological meaning.", ""]
    protocol_diagnosis = config.get("protocol_diagnosis", [
        "The intended protocol was ordinary place reversal: maintain one correct corner during acquisition, switch once to the diagonally opposite corner, and keep that new corner active while the animal relearns.",
        "The controller-recorded targets must be used to determine what was actually executed. Any mismatch from the written schedule is reported as a protocol deviation rather than silently relabelled.",
    ])
    report += ["## Protocol diagnosis", ""]
    for paragraph in protocol_diagnosis:
        report += [paragraph, ""]
    if config.get("interim_findings"):
        report += ["## Interim findings", ""]
        for paragraph in config["interim_findings"]:
            report += [paragraph, ""]
    interpretation_status = config.get("interpretation_status", [
        "This is a descriptive screening report. Treatment identity, light/dark schedule, exact phase-switch timestamps, reward contingencies, and planned stopping criteria require confirmation before inferential analysis.",
    ])
    report += ["## Interpretation status", ""]
    for paragraph in interpretation_status:
        report += [paragraph, ""]
    report += ["## Analysis provenance", "",
               "- Session inventory, visit counts, nosepoke counts, corner use and hardware-event summaries are direct quality-control views of the exported tables.",
               "- Individual/cohort learning curves, visit blocks, terminal accuracy, trials-to-criterion, error decomposition and activity–accuracy separation implement the outcomes pre-specified in `projects/intellicage/plan.md` §8.",
               "- Actograms and IS/IV/RA circadian summaries were motivated by the spontaneous-activity signature discussed from Voikar et al. (2018) in that plan; they are descriptive here because the lighting schedule has not been verified.",
               "- Phase names, the intended 17 July 00:01 reversal, four balanced target corners and the seven-second nosepoke door rule came from the archived learning log/pilot report and NewBehavior place-learning tutorial.",
               "- No values were copied from the earlier AI reports. Plotted values were recomputed from `Visits.txt`, `Nosepokes.txt`, `Animals.txt`, and `HardwareEvents.txt`. Where the earlier narrative conflicts with those tables, the raw export controls the current report.", ""]
    contrasts = _group_comparison(config, loaded, output)
    if not contrasts.empty:
        report += ["## Between-group comparison", "",
                   "Each row is one pre-declared measure with the MOUSE as the experimental unit. "
                   "`p` is an exact two-sided label permutation over all group assignments; the CI is a "
                   "percentile bootstrap on the difference in means. `min_attainable_p` is the smallest "
                   "p-value this design can produce, so a p above it is not evidence of no difference — "
                   "it is a study too small to resolve one. Values behind this table are in "
                   "`group_measures_by_animal.csv` and `group_comparison.csv`.", "",
                   "| Measure | " + " | ".join(contrasts.iloc[0][["group_a", "group_b"]]) +
                   " | Difference | 95% CI | p (exact) | min p |",
                   "|---|---:|---:|---:|:---:|---:|---:|"]
        for _, row in contrasts.iterrows():
            report.append(
                f"| {row['measure']} | {row['mean_a']:.3f} (n={int(row['n_a'])}) | "
                f"{row['mean_b']:.3f} (n={int(row['n_b'])}) | {row['difference']:+.3f} | "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | {row['p_value']:.3f} | "
                f"{row['min_attainable_p']:.3f} |")
        report += ["", "Every confidence interval above includes differences large enough to matter "
                   "biologically, so these data do not establish equivalence between the groups.", ""]
    write_experiment_provenance(output, Path(config_path), stamped)
    # No markdown horizontal rule here: pandoc renders "---" as \rule{}{\linethickness},
    # which this LaTeX stack rejects with "Missing number, treated as zero".
    report += ["## Generation", "",
               f"Generated by neu-intellicage {__version__} from `{Path(config_path).name}`. "
               f"Input hashes and parameters are in `provenance.json` here and in each session folder.", ""]
    path = output / "report.md"; path.write_text("\n".join(report))
    return path
