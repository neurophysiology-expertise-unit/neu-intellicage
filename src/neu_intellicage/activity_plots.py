"""Figures for the activity report.

Every function writes the plotted numbers next to the image, per AGENTS.md: a
figure is a reading aid, and the table beside it is what a reader re-analyses.

Two conventions differ from `plots.py` and are deliberate:

* Time runs in Zeitgeber hours, so the dark phase is one block. See
  `activity.add_zeitgeber` for why a calendar day is the wrong row.
* Actograms are SINGLE plotted by default. Double-plotting exists to keep a
  DRIFTING activity onset continuous across the midnight boundary; under a fixed
  light schedule with no drift it prints every datum twice for nothing and
  halves the width of each cell. `double=True` is there for the manuscript
  convention and for the day a free-run protocol is actually run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

from .activity import LIGHTS_OFF, LIGHTS_ON, add_zeitgeber, domain_of

INK, MUTED, LINE = "#1c1c1e", "#6b6b70", "#c9c9ce"
FAULT = "#b3261e"
GROUP_COLORS = ("#3b5bdb", "#c9560f")   # validated: CVD dE 29.6, normal dE 34.4
ACTIVITY_RAMP = LinearSegmentedColormap.from_list("activity", ["#ffffff", "#101018"])
FAILURE_RAMP = LinearSegmentedColormap.from_list("failure", ["#f7f2f2", "#8c1d18"])


def _light_dark_bar(ax, span: int, day_length: int, y: float = -1.35,
                    height: float = 0.55) -> None:
    """Draw the light/dark reference strip just above a panel."""
    for start in range(0, span, 24):
        for offset, dark in ((0, False), (day_length, True)):
            width = day_length if not dark else 24 - day_length
            ax.add_patch(Rectangle((start + offset, y), width, height, clip_on=False,
                                   zorder=5, lw=.5, edgecolor=MUTED,
                                   facecolor="#2c2c33" if dark else "#ffffff"))


def _hourly_matrix(frame: pd.DataFrame, days, values: str | None = None) -> np.ndarray:
    """Day x ZT-hour grid; missing cells stay NaN rather than becoming zero."""
    if values is None:
        grid = (frame.groupby(["zt_day", "zt_hour"]).size().rename("v").reset_index()
                .pivot(index="zt_day", columns="zt_hour", values="v")
                .reindex(index=days, columns=range(24)).fillna(0.0))
    else:
        grid = (frame.groupby(["zt_day", "zt_hour"])[values].mean().rename("v").reset_index()
                .pivot(index="zt_day", columns="zt_hour", values="v")
                .reindex(index=days, columns=range(24)))
    return grid.to_numpy(dtype=float)


def actogram(session, output: Path, lights_on: int = LIGHTS_ON,
             lights_off: int = LIGHTS_OFF, double: bool = False,
             flag_days: dict[str, list] | None = None) -> pd.DataFrame:
    """Per-animal actogram of visits per ZT hour.

    ``flag_days`` maps a label to a list of dates to mark in the left margin --
    used for days a corner was known to be faulty, so a reader never mistakes a
    hardware outage for behaviour.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x = add_zeitgeber(session.visits if hasattr(session, "visits") else session,
                      lights_on, lights_off)
    animals = sorted(x["AnimalName"].unique())
    days = pd.date_range(x["zt_day"].min(), x["zt_day"].max(), freq="D")
    day_length = (lights_off - lights_on) % 24

    table = (x.groupby(["AnimalName", "zt_day", "zt_hour"]).size().rename("visits")
             .reset_index())
    table.to_csv(output / "actogram_zt_hourly.csv", index=False)

    matrices = {a: _hourly_matrix(x[x["AnimalName"].eq(a)], days) for a in animals}
    vmax = max(1.0, float(np.percentile(np.concatenate([m.ravel() for m in matrices.values()]), 98)))
    span = 48 if double else 24
    flagged = {label: {pd.Timestamp(d).normalize() for d in dates}
               for label, dates in (flag_days or {}).items()}

    fig, axes = plt.subplots(len(animals), 1, figsize=(6.5 * span / 24 + 2.0,
                                                       1.9 * len(animals) + 1.0),
                             squeeze=False)
    for row, animal in enumerate(animals):
        ax = axes[row, 0]
        matrix = matrices[animal]
        if double:
            following = np.vstack([matrix[1:], np.full((1, 24), np.nan)])
            matrix = np.concatenate([matrix, following], axis=1)
        ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ACTIVITY_RAMP,
                  vmin=0, vmax=vmax, extent=(0, span, len(days) - .5, -.5))
        for boundary in range(day_length, span, 12):
            ax.axvline(boundary, color=GROUP_COLORS[0], lw=.9, alpha=.5, zorder=4)
        _light_dark_bar(ax, span, day_length)
        ax.set(xlim=(0, span), ylim=(len(days) - .5, -1.4),
               xticks=range(0, span + 1, 12))
        ax.set_xticklabels(range(0, span + 1, 12), fontsize=8, color=MUTED)
        ax.set_yticks(range(len(days)))
        ax.set_yticklabels([d.strftime("%d %b") for d in days], fontsize=7.5, color=MUTED)
        ax.set_ylabel(animal, fontsize=9.5, color=INK, labelpad=32)
        ax.tick_params(length=2, colors=MUTED, pad=13)
        # x in axes fraction so the flag sits the same distance out at any span
        transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for index, day in enumerate(days):
            if any(day in dates for dates in flagged.values()):
                ax.add_patch(Rectangle((-0.030, index - .36), 0.018, .72, transform=transform,
                                       clip_on=False, facecolor=FAULT, lw=0, zorder=6))
        for spine in ax.spines.values():
            spine.set_color(LINE)
    axes[-1, 0].set_xlabel(f"Zeitgeber hour (ZT0 = lights on {lights_on:02d}:00)",
                           fontsize=9, color=MUTED)
    title = "Visits per hour, one row per ZT day"
    if double:
        title += " — double plotted, right half repeats the next day"
    subtitle = "bar above panel = light / dark"
    if flagged:
        subtitle += "   ·   red flag = " + ", ".join(flagged)
    fig.suptitle(f"{title}\n{subtitle}   ·   scale 0–{vmax:.0f} visits/h",
                 fontsize=11, color=INK)
    fig.subplots_adjust(left=.175, right=.99, top=.93, bottom=.06, hspace=.42)
    fig.savefig(output / ("actogram_zt_double.png" if double else "actogram_zt.png"),
                dpi=170, facecolor="white")
    plt.close(fig)
    return table


def reward_failure(session, output: Path, lights_on: int = LIGHTS_ON,
                   lights_off: int = LIGHTS_OFF,
                   corners=(1, 2, 3, 4)) -> pd.DataFrame:
    """Share of REWARDED visits that delivered no lick, by corner, day and hour.

    This is the figure that separates a hardware fault from behaviour. A broken
    valve fails at every hour of the day and night; an animal losing interest in
    a corner does not. Correctness here is the controller's own
    ``CornerCondition == 1``, so a corner can be scored correct all day while
    paying out nothing -- which is exactly the failure this catches and no
    accuracy measure can.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    visits = session.visits if hasattr(session, "visits") else session
    x = add_zeitgeber(visits, lights_on, lights_off)
    if "CornerCondition" not in x or "LickNumber" not in x:
        raise ValueError("reward_failure needs CornerCondition and LickNumber")
    rewarded = x[x["CornerCondition"].eq(1)].copy()
    rewarded["dry"] = rewarded["LickNumber"].fillna(0).le(0).astype(float)
    days = pd.date_range(x["zt_day"].min(), x["zt_day"].max(), freq="D")
    day_length = (lights_off - lights_on) % 24

    table = (rewarded.groupby(["Corner", "zt_day", "zt_hour"])
             .agg(rewarded_visits=("dry", "size"), dry_visits=("dry", "sum"))
             .reset_index())
    table["dry_share"] = table["dry_visits"] / table["rewarded_visits"]
    table.to_csv(output / "reward_failure_by_corner_hour.csv", index=False)

    fig, axes = plt.subplots(1, len(corners), figsize=(3.6 * len(corners) + 1.4, 4.6),
                             sharey=True, squeeze=False)
    image = None
    for index, corner in enumerate(corners):
        ax = axes[0, index]
        matrix = _hourly_matrix(rewarded[rewarded["Corner"].eq(corner)], days, values="dry")
        ramp = FAILURE_RAMP.copy()
        ramp.set_bad("#e9e9ec")
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ramp,
                          vmin=0, vmax=1, extent=(0, 24, len(days) - .5, -.5))
        _light_dark_bar(ax, 24, day_length)
        ax.axvline(day_length, color=GROUP_COLORS[0], lw=.9, alpha=.5)
        ax.set(xlim=(0, 24), ylim=(len(days) - .5, -1.4), xticks=[0, 12, 24])
        ax.set_xticklabels([0, 12, 24], fontsize=8, color=MUTED)
        ax.set_title(f"Corner {corner}", fontsize=10.5, color=INK, pad=16)
        ax.set_xlabel("Zeitgeber hour", fontsize=9, color=MUTED)
        ax.tick_params(length=2, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(LINE)
    axes[0, 0].set_yticks(range(len(days)))
    axes[0, 0].set_yticklabels([d.strftime("%d %b") for d in days], fontsize=8, color=MUTED)
    bar = fig.colorbar(image, ax=axes[0].tolist(), pad=.028, fraction=.020)
    bar.set_label("Share of correct visits that delivered NO water", fontsize=9, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    bar.outline.set_edgecolor(LINE)
    fig.suptitle("Reward failure by corner and hour — grey = no correct visit in that hour\n"
                 "bar above panel = light / dark", fontsize=11, color=INK, y=.99)
    fig.subplots_adjust(left=.062, right=.895, top=.80, bottom=.13, wspace=.09)
    fig.savefig(output / "reward_failure.png", dpi=170, facecolor="white")
    plt.close(fig)
    return table


def effect_forest(scan: pd.DataFrame, output: Path, floor: float | None = None) -> None:
    """Effect sizes grouped into behavioural domains, largest first.

    Measures are ordered within a domain, never across it, so a reader compares
    like with like. The design floor is drawn as a line rather than left implied:
    at four animals per group nothing can fall below it, and a p sitting ON the
    floor is the strongest result the design can produce, not a strong result.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = scan.dropna(subset=["difference"]).copy()
    if frame.empty:
        return
    if "domain" not in frame:
        frame["domain"] = frame["measure"].map(domain_of)
    frame["effect"] = frame.get("hedges_g", frame["difference"])
    domains = [d for d in frame["domain"].drop_duplicates()]
    heights = [max(1, int((frame["domain"] == d).sum())) for d in domains]
    limit = float(np.nanmax(np.abs(frame["effect"]))) * 1.08 or 1.0
    fig, axes = plt.subplots(len(domains), 1, squeeze=False, sharex=True,
                             figsize=(8.5, 0.26 * sum(heights) + 0.42 * len(domains) + 1.1),
                             gridspec_kw={"height_ratios": heights})
    for ax, domain in zip(axes[:, 0], domains):
        block = frame[frame["domain"].eq(domain)].sort_values("effect", key=abs)
        y = np.arange(len(block))
        at_floor = block["p_value"] <= (floor or 0) + 1e-9
        ax.barh(y, block["effect"], height=.62,
                color=np.where(at_floor, FAULT, "#9a9aa2"), zorder=3)
        ax.axvline(0, color=MUTED, lw=.8, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(block["measure"], fontsize=7.5, color=INK)
        ax.set_ylim(-.7, len(block) - .3)
        ax.set_xlim(-limit, limit)
        ax.set_title(domain, fontsize=9.5, color=INK, loc="left", pad=4)
        ax.tick_params(length=2, colors=MUTED)
        ax.grid(axis="x", color="#ececef", lw=.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(LINE)
    axes[-1, 0].set_xlabel("Hedges' g   (positive = first group higher)", fontsize=9, color=MUTED)
    label = "red = p at the design floor" if floor else "effect size"
    fig.suptitle(f"Activity phenotype scan — {label}", fontsize=11, color=INK)
    fig.subplots_adjust(left=.34, right=.985, top=.94, bottom=.06, hspace=.30)
    fig.savefig(output / "effect_forest.png", dpi=170, facecolor="white")
    plt.close(fig)


def session_consistency(per_session: pd.DataFrame, measures: list[str], output: Path,
                        groups: dict[str, list[str]]) -> None:
    """Per-animal values session by session, one panel per measure.

    This is the figure that answers whether pooling sessions is defensible: a
    measure worth pooling keeps the same group ordering in every session, and
    one that flips is a session-specific fluke however small its pooled p.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (name_a, members_a), (name_b, members_b) = list(groups.items())
    sessions = list(per_session["session"].drop_duplicates())
    columns = min(3, len(measures))
    rows = -(-len(measures) // columns)
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(4.3 * columns, 3.1 * rows))
    positions = np.arange(len(sessions))
    for index, measure in enumerate(measures):
        ax = axes[index // columns, index % columns]
        for members, colour, label in ((members_a, GROUP_COLORS[0], name_a),
                                       (members_b, GROUP_COLORS[1], name_b)):
            block = per_session[per_session["AnimalName"].isin(members)]
            wide = block.pivot_table(index="session", columns="AnimalName", values=measure)
            wide = wide.reindex(sessions)
            for animal in wide.columns:
                ax.plot(positions, wide[animal], color=colour, alpha=.35, lw=1, zorder=2)
            ax.plot(positions, wide.mean(axis=1), color=colour, lw=2.4, marker="o",
                    ms=6, zorder=3, label=label,
                    markeredgecolor="white", markeredgewidth=1.4)
        ax.set_xticks(positions)
        ax.set_xticklabels([s.split()[0] for s in sessions], fontsize=8, color=MUTED)
        ax.set_title(measure, fontsize=9, color=INK, loc="left")
        ax.tick_params(length=2, colors=MUTED, labelsize=8)
        ax.grid(axis="y", color="#ececef", lw=.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(LINE)
        if index == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="best")
    for extra in range(len(measures), rows * columns):
        axes[extra // columns, extra % columns].axis("off")
    fig.suptitle("Is the group difference the same in every session?\n"
                 "thin lines = individual mice, thick = group mean",
                 fontsize=11, color=INK)
    fig.subplots_adjust(left=.075, right=.985, top=.88, bottom=.07, hspace=.42, wspace=.25)
    fig.savefig(output / "session_consistency.png", dpi=170, facecolor="white")
    plt.close(fig)
