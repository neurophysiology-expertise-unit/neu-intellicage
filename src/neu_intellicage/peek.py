"""A one-command daily look at whether the animals are learning.

Built for the question "should I keep this protocol running?", asked once a day
in front of a terminal. It answers with one line per mouse and one figure, and
it decides significance the same way the full report does, so a peek and a
report can never disagree.

The verdict per mouse rests on the exact binomial boundary for that mouse's own
number of choices so far, not on a fixed percentage. Early in a protocol the
honest answer is usually "not yet decidable", and saying so is the point: the
alternative is reading a 50% hit rate on eight moves as learning.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import Session
from .metrics import CHANCE, add_time_fields, chance_boundary, corner_health
from .patrolling import NAIVE_CHANCE, PATROL_CHANCE, detect_task, patrol_cumulative, patrol_daily


def _verdict(correct: int, total: int, chance: float) -> str:
    if total == 0:
        return "no data"
    lower, upper = chance_boundary(total, p0=chance)
    rate = correct / total
    if not np.isnan(upper) and rate > upper:
        return "ABOVE chance"
    if not np.isnan(lower) and rate < lower:
        return "BELOW chance"
    if np.isnan(upper):
        return "not yet decidable"
    return "not distinguishable"


def _cumulative_place(session: Session) -> pd.DataFrame:
    x = add_time_fields(session.visits)
    x = x[x["conditioned"]].sort_values(["AnimalName", "Start"]).copy()
    x["choice_number"] = x.groupby("AnimalName").cumcount() + 1
    x["cumulative_correct"] = x.groupby("AnimalName")["correct"].cumsum()
    x["cumulative_hit_rate"] = x["cumulative_correct"] / x["choice_number"]
    x["expected_by_chance"] = x["choice_number"] * CHANCE
    x["excess_over_chance"] = x["cumulative_correct"] - x["expected_by_chance"]
    return x


def peek(session: Session, output: Path | None = None) -> tuple[str, pd.DataFrame]:
    """Return a printable summary and the per-animal table behind it."""
    task = detect_task(session.visits)
    if task == "unconditioned":
        return ("No corner condition was active in this session, so there is no "
                "learning measure to report. Use the QC views instead.", pd.DataFrame())

    if task == "patrolling":
        chance, denominator = PATROL_CHANCE, "moves"
        cumulative = patrol_cumulative(session.visits)
        daily = patrol_daily(session.visits)
        totals = cumulative.groupby("AnimalName").agg(
            choices=("move_number", "max"), correct=("cumulative_correct", "max"),
            excess=("excess_over_chance", "last")).reset_index()
        extra = daily.groupby("AnimalName").agg(
            longest_run=("longest_run", "max"), laps=("laps_completed", "sum"),
            naive_rate=("hit_rate_all_visits", "mean")).reset_index()
        totals = totals.merge(extra, on="AnimalName", how="left")
    else:
        chance, denominator = CHANCE, "conditioned visits"
        cumulative = _cumulative_place(session)
        totals = cumulative.groupby("AnimalName").agg(
            choices=("choice_number", "max"), correct=("cumulative_correct", "max"),
            excess=("excess_over_chance", "last")).reset_index()

    totals["hit_rate"] = totals["correct"] / totals["choices"]
    bounds = [chance_boundary(int(n), p0=chance) for n in totals["choices"]]
    totals["chance_lower"] = [b[0] for b in bounds]
    totals["chance_upper"] = [b[1] for b in bounds]
    totals["verdict"] = [_verdict(int(c), int(n), chance)
                         for c, n in zip(totals["correct"], totals["choices"])]
    totals["task"] = task
    totals["chance"] = chance

    days = add_time_fields(session.visits)["date"]
    health = corner_health(session.visits)
    alarm = []
    if not health.empty and (health["status"] != "ok").any():
        alarm = ["", "!! HARDWARE WARNING -- check the cage before reading anything below."]
        for _, row in health[health["status"] != "ok"].iterrows():
            if row["status"] in ("FAILED", "degraded"):
                alarm.append(f"   Corner {int(row['Corner'])}: {row['status']} -- "
                             f"{row['dry_rate_recent']:.0%} of correct visits in the last 48 h "
                             f"delivered NO water ({int(row['rewarded_visits_recent'])} rewarded "
                             f"visits). The animals are being extinguished on this corner.")
            else:
                alarm.append(f"   Corner {int(row['Corner'])}: under-visited -- "
                             f"{row['visit_share']:.0%} of all visits against an expected "
                             f"{1 / len(health):.0%}. Check that it can be entered.")
        alarm.append("")
    header = alarm + [
        f"Task detected from the export: {task.upper()}"
        + (" (clockwise 1->2->3->4, target advances only on a hit)" if task == "patrolling" else ""),
        f"Recorded {days.nunique()} day(s): {days.min()} to {days.max()}",
        f"Chance = {chance:.3f} over {denominator}."
        + ("  A mouse that has learned only 'do not re-enter the corner I just left' "
           "already scores 1/3, which is why 1/4 is the wrong reference here."
           if task == "patrolling" else ""),
        "",
        f"{'Mouse':<10}{'hits':>6}{'/':^3}{denominator[:7]:<8}{'rate':>7}{'need':>7}{'excess':>8}  verdict",
    ]
    lines = []
    for _, row in totals.iterrows():
        need = "n/a" if np.isnan(row["chance_upper"]) else f"{row['chance_upper']:.2f}"
        lines.append(f"{row['AnimalName']:<10}{int(row['correct']):>6}{'/':^3}"
                     f"{int(row['choices']):<8}{row['hit_rate']:>7.3f}{need:>7}"
                     f"{row['excess']:>+8.1f}  {row['verdict']}")
    footer = ["",
              "'need' is the hit rate this mouse would have to exceed, given the number of",
              "choices it has actually made, for the result to be above chance at alpha=0.05.",
              "'excess' is hits beyond what chance predicts over the same choices.",
              ]
    if task == "patrolling":
        footer += [
            "",
            f"{'Mouse':<10}{'longest run':>12}{'laps':>6}{'rate vs 1/4':>13}",
        ]
        for _, row in totals.iterrows():
            footer.append(f"{row['AnimalName']:<10}{int(row['longest_run']):>12}"
                          f"{int(row['laps']):>6}{row['naive_rate']:>13.3f}")
        footer += ["",
                   "A 'lap' is four consecutive correct corners: one full circuit of the cage.",
                   f"'rate vs 1/4' is the all-visits hit rate against the naive {NAIVE_CHANCE} "
                   "reference; it flatters the animal and is shown only for comparison."]

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        totals.to_csv(output / "peek_summary.csv", index=False)
        if not health.empty:
            health.to_csv(output / "peek_corner_health.csv", index=False)
        cumulative.to_csv(output / "peek_cumulative.csv", index=False)
        _plot(cumulative, totals, task, chance, denominator, output)

    return "\n".join(header + lines + footer), totals


def _plot(cumulative: pd.DataFrame, totals: pd.DataFrame, task: str,
          chance: float, denominator: str, output: Path) -> None:
    number = "move_number" if task == "patrolling" else "choice_number"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # Left: the cumulative record. Slope is the learning rate; the dashed line is
    # what a non-learning mouse produces, so the vertical gap is the evidence.
    for animal, frame in cumulative.groupby("AnimalName"):
        axes[0].plot(frame[number], frame["cumulative_correct"], lw=1.6, label=animal)
    span = np.arange(0, int(cumulative[number].max()) + 1)
    axes[0].plot(span, span * chance, ls="--", color="0.35", lw=1.4,
                 label=f"chance ({chance:.2f})")
    axes[0].set(xlabel=f"{denominator.capitalize()} made", ylabel="Cumulative correct",
                title="Cumulative record — slope is the learning rate")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    # Right: running hit rate against the boundary that shrinks as evidence grows.
    for animal, frame in cumulative.groupby("AnimalName"):
        axes[1].plot(frame[number], frame["cumulative_hit_rate"], lw=1.4, label=animal)
    grid = np.arange(1, int(cumulative[number].max()) + 1)
    upper = [chance_boundary(int(n), p0=chance)[1] for n in grid]
    axes[1].plot(grid, upper, color="0.35", ls="--", lw=1.4, label="above-chance boundary")
    axes[1].axhline(chance, color="0.6", ls=":", lw=1)
    axes[1].set(xlabel=f"{denominator.capitalize()} made", ylabel="Running hit rate",
                ylim=(0, 1), title="Cross the dashed line and stay above it")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(f"{task.capitalize()} — daily peek")
    fig.tight_layout()
    fig.savefig(output / "peek.png", dpi=180)
    plt.close(fig)
