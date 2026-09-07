"""Build the spontaneous-activity report across several sessions.

Kept separate from `report.build_experiment_report` on purpose. That report is
about whether the animals learned; this one is about how they behave when
nobody is asking them to learn anything. The two answer different questions,
have different validity conditions, and should be readable independently -- a
learning result that depends on a training protocol surviving is not comparable
to an activity result that does not.

The report is config-driven so that a run is reproducible from a file rather
than from an argument list someone typed once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import activity, activity_plots
from .groups import (benjamini_hochberg, days_to_separation, exact_permutation_p,
                     hedges_g, scan_profile, select_headline_measures,
                     session_interaction_p)
from .io import load_session
from .provenance import write_provenance

FLOOR_TOLERANCE = 1e-9


def _load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("sessions", "groups"):
        if key not in config:
            raise ValueError(f"activity report config needs a '{key}' entry")
    if len(config["groups"]) != 2:
        raise ValueError("activity report compares exactly two groups")
    return config


def _check_group_labels(sessions: dict, groups: dict[str, list[str]]) -> list[str]:
    """Describe how the configured grouping lines up with the export's GroupName.

    AGENTS.md forbids silently reinterpreting group labels. The configured
    mapping wins -- it comes from the cohort sheet -- but the correspondence has
    to be printed, because the export's own names carry no information about
    which arm is which and can suggest the opposite of the truth. On this cohort
    the knockdown animals are labelled `Control` in the export, so a reader who
    trusted `GroupName` would read every result backwards.

    A configured group that spans more than one export label is flagged
    additionally: that is a genuine ambiguity, not just a naming clash.
    """
    lines = []
    for label, session in sessions.items():
        animals = getattr(session, "animals", None)
        if animals is None or "GroupName" not in animals:
            continue
        export = animals.dropna(subset=["GroupName"]).set_index("AnimalName")["GroupName"]
        parts, straddles = [], []
        for name, members in groups.items():
            present = [m for m in members if m in export.index]
            if not present:
                continue
            observed = sorted(set(export.loc[present].astype(str)))
            parts.append(f"`{name}` = GroupName {observed}")
            if len(observed) > 1:
                straddles.append(name)
        if not parts:
            continue
        suffix = f"  **AMBIGUOUS: {', '.join(straddles)} spans more than one label.**" if straddles else ""
        lines.append(f"{label}: " + "; ".join(parts) + suffix)
    return lines


def build_activity_report(config_path: str | Path, output: str | Path) -> str:
    config = _load_config(config_path)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    lights_on = int(config.get("lights_on", activity.LIGHTS_ON))
    lights_off = int(config.get("lights_off", activity.LIGHTS_OFF))
    groups: dict[str, list[str]] = config["groups"]
    (name_a, members_a), (name_b, members_b) = list(groups.items())
    root = Path(config.get("sessions_root", "."))

    # Days the cage was not measuring what it claims to -- a dead corner changes
    # every animal's visit rate and regularity, so an activity measure taken
    # then is about the hardware, not the mouse.
    exclude_days = list(config.get("exclude_days") or [])
    sessions, daily, coverage = {}, {}, []
    for entry in config["sessions"]:
        label, path = entry["label"], root / entry["path"]
        session = load_session(path)
        sessions[label] = session
        table = activity.daily_measures(session, lights_on, lights_off,
                                        exclude_days=exclude_days)
        daily[label] = table
        span = activity.add_zeitgeber(session.visits, lights_on, lights_off)
        coverage.append({
            "session": label, "visits": len(session.visits),
            "first": span["Start"].min().strftime("%Y-%m-%d %H:%M"),
            "last": span["Start"].max().strftime("%Y-%m-%d %H:%M"),
            "complete_zt_days": 0 if table.empty else int(table["zt_day"].nunique()),
        })
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(output / "tables" / "session_coverage.csv", index=False)

    usable = {label: table for label, table in daily.items() if not table.empty}
    if not usable:
        raise ValueError("no session contributed a complete Zeitgeber day; "
                         "every session is shorter than 24 h of covered time")

    # ---- pooled scan -------------------------------------------------------
    pooled = activity.pool_sessions(usable, centre=bool(config.get("centre_days", True)))
    centred = activity.pooled_wide(pooled, "value")
    raw = activity.pooled_wide(pooled, "raw")
    measures = activity.measure_columns(centred)
    scan = scan_profile(centred, groups, measures=measures)
    scan = select_headline_measures(scan, centred)
    scan["domain"] = scan["measure"].map(activity.domain_of)
    scan["note"] = scan["measure"].map(activity.describe)
    raw_indexed = raw.set_index("AnimalName")
    scan[f"mean_{name_a}_raw"] = [float(raw_indexed.loc[members_a, m].mean()) for m in scan["measure"]]
    scan[f"mean_{name_b}_raw"] = [float(raw_indexed.loc[members_b, m].mean()) for m in scan["measure"]]
    scan["hedges_g"] = [hedges_g(raw_indexed.loc[members_a, m].to_numpy(),
                                 raw_indexed.loc[members_b, m].to_numpy())
                        for m in scan["measure"]]
    floor = float(scan["min_attainable_p"].iloc[0]) if len(scan) else float("nan")
    scan.sort_values("p_value").to_csv(output / "tables" / "pooled_scan.csv", index=False)
    raw.to_csv(output / "tables" / "pooled_profile_raw.csv", index=False)

    # ---- per-session consistency and interaction ---------------------------
    at_floor = scan[scan["p_value"] <= floor + FLOOR_TOLERANCE]["measure"].tolist()
    primary = list(config.get("primary_measures") or [])
    followed = primary + [m for m in at_floor if m not in primary]
    per_session_rows, interactions = [], []
    for label, table in usable.items():
        for column in followed:
            measure, _, phase = column.rpartition("_")
            block = table[table["measure"].eq(measure) & table["phase"].eq(phase)]
            if block.empty:
                continue
            for animal, value in block.groupby("AnimalName")["value"].mean().items():
                per_session_rows.append({"AnimalName": animal, "session": label,
                                         "measure": column, "value": float(value)})
    per_session = pd.DataFrame(per_session_rows)
    session_table = []
    if not per_session.empty:
        wide_sessions = per_session.pivot_table(index=["AnimalName", "session"],
                                                columns="measure", values="value").reset_index()
        wide_sessions.to_csv(output / "tables" / "per_session_measures.csv", index=False)
        for column in followed:
            if column not in wide_sessions:
                continue
            for label in usable:
                block = wide_sessions[wide_sessions["session"].eq(label)].set_index("AnimalName")
                if column not in block:
                    continue
                a = block[column].reindex(members_a).to_numpy(dtype=float)
                b = block[column].reindex(members_b).to_numpy(dtype=float)
                if np.isnan(a).any() or np.isnan(b).any():
                    continue
                p, _, _ = exact_permutation_p(a, b)
                margin = float(max(b.min() - a.max(), a.min() - b.max()))
                session_table.append({
                    "measure": column, "session": label,
                    "complete_days": int(usable[label]["zt_day"].nunique()),
                    f"mean_{name_a}": float(a.mean()), f"mean_{name_b}": float(b.mean()),
                    "difference": float(a.mean() - b.mean()), "hedges_g": hedges_g(a, b),
                    "p_value": p, "separated": margin > 0})
            result = session_interaction_p(
                per_session[per_session["measure"].eq(column)].rename(columns={"value": column}),
                column, groups)
            interactions.append({"measure": column, "interaction_p": result["p_value"],
                                 "spread": result["observed_spread"],
                                 "note": result.get("note", "")})
    session_table = pd.DataFrame(session_table)
    if not session_table.empty:
        session_table.to_csv(output / "tables" / "per_session_contrast.csv", index=False)
    interactions = pd.DataFrame(interactions)
    if not interactions.empty:
        interactions.to_csv(output / "tables" / "session_interaction.csv", index=False)

    # ---- how many days does the contrast need? -----------------------------
    longest = max(usable, key=lambda label: usable[label]["zt_day"].nunique())
    curves = []
    for column in followed:
        measure, _, phase = column.rpartition("_")
        curve = days_to_separation(usable[longest], measure, phase, groups)
        if not curve.empty:
            curves.append(curve.assign(measure=column, session=longest))
    curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if not curves.empty:
        curves.to_csv(output / "tables" / "days_to_separation.csv", index=False)

    # ---- figures -----------------------------------------------------------
    flag_days = config.get("flag_days") or {}
    figures = output / "figures"
    actogram_session = config.get("actogram_session", longest)
    if actogram_session in sessions:
        activity_plots.actogram(sessions[actogram_session], figures, lights_on, lights_off,
                                double=bool(config.get("double_plot", False)),
                                flag_days=flag_days)
        try:
            activity_plots.reward_failure(sessions[actogram_session], figures,
                                          lights_on, lights_off)
        except ValueError:
            pass
    activity_plots.effect_forest(scan, figures, floor=floor)
    if not per_session.empty and followed:
        shown = [c for c in followed if c in wide_sessions][:6]
        if shown:
            activity_plots.session_consistency(wide_sessions, shown, figures, groups)

    text = _render(config, coverage, scan, session_table, interactions, curves,
                   groups, floor, longest, _check_group_labels(sessions, groups))
    (output / "report.md").write_text(text, encoding="utf-8")
    write_provenance(output, root, {
        "command": "activity-report", "config": str(config_path),
        "lights_on": lights_on, "lights_off": lights_off,
        "sessions": [entry["path"] for entry in config["sessions"]],
        "groups": groups, "centre_days": bool(config.get("centre_days", True)),
        "measures_scanned": len(scan), "design_floor_p": floor,
        "excluded_days": exclude_days,
    })
    return text


def _codes(names: list[str], prefix: str) -> dict[str, str]:
    """Short row codes so a long measure name cannot push the numbers out of
    their column. The full name is restated in a legend under each table."""
    return {name: f"{prefix}{index}" for index, name in enumerate(names, start=1)}


def _table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> list[str]:
    """Hand-rolled markdown table: ``columns`` is (source, header, format)."""
    present = [(source, header, spec) for source, header, spec in columns if source in frame]
    lines = ["| " + " | ".join(header for _, header, _ in present) + " |",
             "|" + "|".join("---" for _ in present) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for source, _, spec in present:
            value = row[source]
            if spec == "s":
                cells.append(str(value) if pd.notna(value) else "")
            elif value is None or (isinstance(value, float) and np.isnan(value)):
                cells.append("--")
            else:
                cells.append(format(value, spec))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _legend(codes: dict[str, str], describe=None) -> list[str]:
    lines = [""]
    for name, code in codes.items():
        note = f" — {describe(name)}" if describe else ""
        lines.append(f"- **{code}** = `{name}`{note}")
    return lines + [""]


def _fmt(value: float, places: int = 3) -> str:
    return "--" if value is None or (isinstance(value, float) and np.isnan(value)) \
        else f"{value:.{places}f}"


def _render(config, coverage, scan, session_table, interactions, curves,
            groups, floor, longest, label_warnings) -> str:
    (name_a, members_a), (name_b, members_b) = list(groups.items())
    lines = [f"# Activity phenotype — {config.get('name', 'unnamed')}", ""]
    lines += [
        "This report describes SPONTANEOUS behaviour: how much the mice move between",
        "corners, how their activity is distributed across the light cycle, and how it is",
        "organised in time. It says nothing about learning, and nothing here depends on a",
        "training protocol having worked.", "",
        "## What was measured", "",
        f"Zeitgeber time with lights on at {config.get('lights_on', activity.LIGHTS_ON):02d}:00, so a day",
        "runs lights-on to lights-on and the dark phase is one contiguous block. Only ZT days",
        "whose full 24 hours lie inside the recorded span are used; partial first and last days",
        "are dropped rather than averaged in, because a partial day is not a quiet day.", "",
        *_table(coverage, [("session", "Session", "s"), ("visits", "Visits", ",d"),
                          ("first", "First visit", "s"), ("last", "Last visit", "s"),
                          ("complete_zt_days", "Complete ZT days", "d")]), "",
    ]
    excluded = list(config.get("exclude_days") or [])
    if excluded:
        lines += ["> **Days excluded.** " + ", ".join(excluded) + ".",
                  "> These ZT days are dropped for every animal, so the groups still rest on the",
                  "> same days. They are excluded because the cage was not measuring what it claims",
                  "> to -- a corner that stops delivering water changes how often and how regularly",
                  "> every mouse visits, and no learning measure would notice.", ""]
    if label_warnings:
        lines += ["### Group labels", "",
                  "Every test below uses the grouping given in the config, which comes from the",
                  "cohort sheet. The export's own `GroupName` column is recorded here rather than",
                  "used, because its names do not identify the arms and can suggest the opposite",
                  "of the truth:", ""]
        lines += [f"- {warning}" for warning in label_warnings]
        lines += ["", "Read `GroupName` straight from the export and the direction of every result",
                  "below reverses.", ""]

    lines += ["## The design floor", "",
              f"With {len(members_a)} and {len(members_b)} animals there are only",
              f"{int(scan['permutations'].iloc[0]) if len(scan) else 0} ways to split the cohort, so the",
              f"smallest p this design can produce is **{_fmt(floor, 4)}**. A measure sitting exactly on",
              "that value is the strongest possible result, not a strong one, and no amount of extra",
              "RECORDING can move it -- only extra ANIMALS can. Every p below should be read against",
              "that ceiling.", ""]

    at_floor = scan[scan["p_value"] <= floor + FLOOR_TOLERANCE]
    lines += ["## Pooled scan", "",
              f"{len(scan)} measures, pooled across the sessions that contributed a complete day.",
              "Each day is centred on that day's cohort mean before averaging, so the large",
              "between-session shifts in overall activity -- which both groups share -- cannot drive",
              "the contrast. Centring subtracts the same number from every animal on a day, so it",
              "leaves each day's group difference untouched and changes only how days are weighted.",
              f"Group means are quoted on the RAW scale; the difference is the same either way.", "",
              f"**{len(at_floor)} of {len(scan)} measures reach the floor of {_fmt(floor, 4)}.**", ""]
    top = scan.sort_values("p_value").head(int(config.get("scan_rows", 15))).copy()
    measure_codes = _codes(list(top["measure"]), "M")
    top["code"] = top["measure"].map(measure_codes)
    top["flag"] = np.where(top["headline"], "yes", "")
    lines += _table(top, [
        ("code", "", "s"), ("domain", "Domain", "s"),
        (f"mean_{name_a}_raw", name_a, ".3f"), (f"mean_{name_b}_raw", name_b, ".3f"),
        ("difference", "Diff", "+.3f"), ("hedges_g", "g", "+.2f"),
        ("p_value", "p", ".4f"), ("p_adjusted_bh", "q (BH)", ".3f"),
        ("flag", "Headline", "s"), ("redundant_with", "Restates", "s")])
    lines += _legend(measure_codes, activity.describe)
    lines += ["",
              f"Full table: `tables/pooled_scan.csv` (all {len(scan)} measures).", "",
              "**Multiplicity.** `p_adjusted_bh` is the Benjamini-Hochberg FDR across the whole scan.",
              "Nothing can survive it here and that is arithmetic, not a result: the smallest",
              f"attainable raw p is {_fmt(floor, 4)}, so over {len(scan)} measures the smallest attainable",
              f"q is about {_fmt(floor * len(scan) / max(1, len(at_floor)), 3)}. **This scan is",
              "hypothesis-generating.** Its job is to name the measure that a properly powered study",
              "should pre-specify, not to establish anything on its own.", "",
              "`headline` marks measures kept after dropping near-duplicates; `redundant_with` names",
              "the measure a dropped one restates. This is a reading order, not a correction -- the",
              "p-values and the FDR are computed over the full scan either way.", "",
              "![Effect sizes by domain](figures/effect_forest.png)", ""]

    if not session_table.empty:
        lines += ["## Is it the same in every session?", "",
                  "A measure worth pooling keeps its direction session to session. One that flips is a",
                  "session-specific fluke however small its pooled p.", "",
                  ]
        shown_codes = _codes(list(session_table["measure"].drop_duplicates()), "P")
        block = session_table.copy()
        block["code"] = block["measure"].map(shown_codes)
        block["sep"] = np.where(block["separated"], "yes", "")
        lines += _table(block, [
            ("code", "", "s"), ("session", "Session", "s"),
            ("complete_days", "Days", "d"),
            (f"mean_{name_a}", name_a, ".3f"), (f"mean_{name_b}", name_b, ".3f"),
            ("difference", "Diff", "+.3f"), ("hedges_g", "g", "+.2f"),
            ("p_value", "p", ".4f"), ("sep", "Separated", "s")])
        lines += _legend(shown_codes, activity.describe)
        lines += ["",
                  "![Session consistency](figures/session_consistency.png)", ""]
    if not interactions.empty:
        lines += ["The interaction test asks whether the group difference itself changes between",
                  "sessions. The animal label is permuted once and applied to every session at",
                  "the same time, because the alternative -- permuting each session separately --",
                  "would test a null in which a mouse can change group between sessions.", "",
                  ]
        inter = interactions.copy()
        inter["code"] = inter["measure"].map(lambda m: shown_codes.get(m, m))
        lines += _table(inter, [("code", "", "s"), ("spread", "Spread of diffs", ".3f"),
                                ("interaction_p", "Interaction p", ".4f"),
                                ("note", "Note", "s")])
        lines += ["",
                  "A large p licenses pooling. It is weak evidence: this test shares the same",
                  f"{_fmt(floor, 4)} floor and has less power than the main contrast, so failing to",
                  "detect heterogeneity is not a demonstration that there is none.", ""]

    if not curves.empty:
        lines += [f"## How many days does the contrast need? ({longest})", "",
                  "The contrast re-run on the first k days. This is about measurement noise per",
                  "animal, not about the design: the same eight mice appear every day, so no length",
                  "of recording can lower the floor. What it shows is whether a measure has settled.", ""]
        pivot = curves.pivot_table(index="measure", columns="days", values="separated")
        # pivot_table returns floats, so the booleans have to be mapped back by value
        display = pivot.apply(lambda column: column.map(
            lambda value: "" if pd.isna(value) else ("sep" if value >= 0.5 else "."))).reset_index()
        curve_codes = _codes(list(display["measure"]), "P")
        display["code"] = display["measure"].map(curve_codes)
        spec = [("code", "", "s")] + [(day, str(day), "s") for day in pivot.columns]
        lines += _table(display, spec)
        lines += _legend(curve_codes, activity.describe)
        lines += ["",
                  "`sep` = the two groups' ranges do not overlap on the first k days.", ""]

    lines += ["## What this does and does not establish", "",
              "- A measure at the floor means the four animals of one group are ranked entirely",
              "  above or below the other four. That is the most this cohort can show.",
              "- It cannot separate a GROUP effect from four animals that happen to differ. Only a",
              "  larger cohort can, and the same eight mice re-measured for longer cannot.",
              "- The consistency and interaction tables are the useful output of this report: they",
              "  say which single measure is worth pre-specifying next time. A measure named in",
              "  advance is tested once, so it carries no multiplicity penalty and its raw p stands.", ""]
    return "\n".join(lines)
