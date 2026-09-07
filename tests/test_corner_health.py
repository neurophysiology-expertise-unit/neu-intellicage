import numpy as np
import pandas as pd

from neu_intellicage.metrics import corner_health


def visits(dry_corner=None, dry_rate=0.0, missing_corner=None):
    rows, vid = [], 0
    start = pd.Timestamp("2026-01-01 00:00")
    for step in range(400):
        corner = (step % 4) + 1
        if corner == missing_corner and step % 5 != 0:
            continue          # that corner is hard to enter, but not impossible
        dry = corner == dry_corner and (step % 100) < dry_rate * 100
        rows.append({"VisitID": vid, "AnimalName": "A", "GroupName": "G",
                     "Start": start + pd.Timedelta(minutes=5 * step),
                     "End": start + pd.Timedelta(minutes=5 * step, seconds=20),
                     "Corner": corner, "CornerCondition": 1, "PlaceError": 0,
                     "LickNumber": 0 if dry else 40})
        vid += 1
    return pd.DataFrame(rows)


def test_a_healthy_cage_reports_all_corners_ok():
    health = corner_health(visits())
    assert (health["status"] == "ok").all()
    assert health["dry_rate_overall"].max() == 0.0


def test_a_corner_that_stops_paying_out_is_flagged_failed():
    """The 3 September fault: correct visits scored correct but gave no water."""
    health = corner_health(visits(dry_corner=3, dry_rate=0.9))
    row = health.set_index("Corner").loc[3]
    assert row["status"] == "FAILED"
    assert row["dry_rate_recent"] >= 0.5
    assert (health.set_index("Corner").drop(3)["status"] == "ok").all()


def test_a_partly_failing_corner_is_flagged_degraded_not_failed():
    health = corner_health(visits(dry_corner=2, dry_rate=0.3))
    assert health.set_index("Corner").loc[2, "status"] == "degraded"


def test_a_corner_that_cannot_be_entered_is_flagged_under_visited():
    """The 29-30 August fault: visits to one corner collapsed."""
    health = corner_health(visits(missing_corner=2))
    row = health.set_index("Corner").loc[2]
    assert row["status"] == "under-visited"
    assert row["visit_share_ratio"] < 0.6


def test_a_corner_with_no_visits_at_all_still_appears():
    """A total failure produces no rows; grouping by observed corners would hide it."""
    frame = visits()
    health = corner_health(frame[frame["Corner"].ne(2)])
    assert 2 in set(health["Corner"])
    assert health.set_index("Corner").loc[2, "status"] == "NO VISITS"


def test_missing_lick_column_returns_empty_rather_than_guessing():
    frame = visits().drop(columns=["LickNumber"])
    assert corner_health(frame).empty
