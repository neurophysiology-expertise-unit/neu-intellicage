"""Tests for the activity figures and the report builder."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neu_intellicage import activity_plots
from neu_intellicage.report_activity import _check_group_labels, build_activity_report
from test_activity import ANIMALS, GROUPS, make_visits


def make_session(visits=None, group_names=None):
    visits = make_visits(days=4) if visits is None else visits
    animals = pd.DataFrame({
        "AnimalName": ANIMALS,
        "GroupName": group_names or (["KD"] * 4 + ["Scr"] * 4),
    })
    return SimpleNamespace(visits=visits, animals=animals, nosepokes=pd.DataFrame(),
                           hardware_events=pd.DataFrame())


def test_actogram_writes_the_numbers_beside_the_picture(tmp_path):
    table = activity_plots.actogram(make_session(), tmp_path)
    assert (tmp_path / "actogram_zt.png").exists()
    assert (tmp_path / "actogram_zt_hourly.csv").exists()
    # AGENTS.md: every plotted quantity is also a machine-readable table.
    saved = pd.read_csv(tmp_path / "actogram_zt_hourly.csv")
    assert set(saved.columns) == {"AnimalName", "zt_day", "zt_hour", "visits"}
    assert saved["visits"].sum() == len(table.index.map(lambda _: 0)) * 0 + table["visits"].sum()


def test_double_plot_is_opt_in_and_writes_a_separate_file(tmp_path):
    activity_plots.actogram(make_session(), tmp_path)
    activity_plots.actogram(make_session(), tmp_path, double=True)
    assert (tmp_path / "actogram_zt.png").exists()
    assert (tmp_path / "actogram_zt_double.png").exists()


def test_reward_failure_finds_a_dead_corner(tmp_path):
    visits = make_visits(days=4)
    visits["CornerCondition"] = 1
    # Corner 3 stops paying out halfway through, at every hour of day and night.
    broke = visits["Corner"].eq(3) & visits["Start"].ge(pd.Timestamp("2026-08-13 07:00"))
    visits.loc[broke, "LickNumber"] = 0
    visits.loc[~broke, "LickNumber"] = 20
    table = activity_plots.reward_failure(make_session(visits), tmp_path)
    assert (tmp_path / "reward_failure.png").exists()
    late = table[table["zt_day"].ge(pd.Timestamp("2026-08-13"))]
    dead = late[late["Corner"].eq(3)]["dry_share"]
    alive = late[late["Corner"].ne(3)]["dry_share"]
    assert dead.min() == 1.0 and alive.max() == 0.0


def test_reward_failure_refuses_data_it_cannot_judge(tmp_path):
    visits = make_visits(days=2).drop(columns=["LickNumber"])
    with pytest.raises(ValueError, match="LickNumber"):
        activity_plots.reward_failure(make_session(visits), tmp_path)


def test_group_label_correspondence_is_always_printed():
    # The export calls animals 1-4 "Control"; the cohort sheet says they are the
    # knockdowns. A clean 1:1 mapping is still reported, because the export's
    # NAME points the wrong way even when the split is unambiguous.
    session = make_session(group_names=["Control"] * 4 + ["Treatment"] * 4)
    lines = _check_group_labels({"S1": session}, GROUPS)
    assert len(lines) == 1
    assert "'KD'" in lines[0] or "`KD`" in lines[0]
    assert "Control" in lines[0] and "Treatment" in lines[0]
    assert "AMBIGUOUS" not in lines[0]


def test_a_group_spanning_two_export_labels_is_flagged_ambiguous():
    session = make_session(group_names=["Control"] * 3 + ["Treatment"] * 5)
    lines = _check_group_labels({"S1": session}, GROUPS)
    assert "AMBIGUOUS" in lines[0] and "KD" in lines[0]


def test_a_session_without_group_names_is_skipped():
    session = make_session()
    session.animals = session.animals.drop(columns=["GroupName"])
    assert _check_group_labels({"S1": session}, GROUPS) == []


def _write_session(root, name, visits):
    cage = root / name / "IntelliCage"
    cage.mkdir(parents=True)
    tags = {animal: 900_000_000_000 + index for index, animal in enumerate(ANIMALS)}
    frame = visits.copy()
    frame["AnimalTag"] = frame["AnimalName"].map(tags)
    frame["Cage"] = 1
    frame[["VisitID", "AnimalTag", "Start", "End", "Cage", "Corner",
           "CornerCondition", "PlaceError", "LickNumber", "PresenceNumber",
           "PresenceDuration"]].to_csv(cage / "Visits.txt", sep="\t", index=False)
    # Animals.txt sits beside the IntelliCage directory, not inside it, and its
    # five columns are positional -- see io.load_session.
    pd.DataFrame({"AnimalName": ANIMALS, "AnimalTag": list(tags.values()),
                  "Sex": ["M"] * 8,
                  "GroupName": ["Control"] * 4 + ["Treatment"] * 4,
                  "AnimalNotes": [""] * 8}).to_csv(root / name / "Animals.txt",
                                                   sep="\t", index=False)


def test_report_runs_end_to_end_and_names_the_floor(tmp_path):
    root = tmp_path / "sessions"
    bias = {a: (0.55 if a in GROUPS["KD"] else 0.70) for a in ANIMALS}
    _write_session(root, "one", make_visits(start="2026-08-11 07:00", days=4,
                                            per_hour=10, dark_bias=bias))
    _write_session(root, "two", make_visits(start="2026-08-20 07:00", days=5,
                                            per_hour=10, dark_bias=bias))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "name": "synthetic", "sessions_root": str(root),
        "sessions": [{"label": "S1 first", "path": "one"},
                     {"label": "S2 second", "path": "two"}],
        "groups": {"KD": GROUPS["KD"], "Scr": GROUPS["Scr"]},
        "lights_on": 7, "lights_off": 19, "actogram_session": "S2 second",
    }), encoding="utf-8")

    text = build_activity_report(config, tmp_path / "out")
    out = tmp_path / "out"
    assert (out / "report.md").exists() and (out / "provenance.json").exists()
    for name in ("pooled_scan.csv", "session_coverage.csv", "per_session_contrast.csv",
                 "session_interaction.csv", "days_to_separation.csv"):
        assert (out / "tables" / name).exists(), name
    assert (out / "figures" / "effect_forest.png").exists()

    assert "0.0286" in text                      # the design floor is stated
    assert "hypothesis-generating" in text
    # The configured grouping disagrees with the export's GroupName, and the
    # report must say so rather than quietly using one of them.
    assert "GroupName" in text

    scan = pd.read_csv(out / "tables" / "pooled_scan.csv")
    planted = scan.set_index("measure").loc["dark_phase_visit_fraction_all"]
    assert planted["p_value"] == pytest.approx(2 / 70)
    assert planted["difference"] < 0           # KD less nocturnal, as planted

    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["parameters"]["design_floor_p"] == pytest.approx(2 / 70)


def test_report_refuses_sessions_too_short_to_yield_a_day(tmp_path):
    root = tmp_path / "sessions"
    short = make_visits(days=2)
    short = short[short["Start"] < pd.Timestamp("2026-08-11 20:00")]
    _write_session(root, "brief", short)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "name": "too short", "sessions_root": str(root),
        "sessions": [{"label": "S1", "path": "brief"}],
        "groups": {"KD": GROUPS["KD"], "Scr": GROUPS["Scr"]},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="complete Zeitgeber day"):
        build_activity_report(config, tmp_path / "out")


def test_config_must_name_exactly_two_groups(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "sessions": [], "groups": {"only": ANIMALS}}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two groups"):
        build_activity_report(config, tmp_path / "out")
