from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_fields(visits: pd.DataFrame) -> pd.DataFrame:
    out = visits.copy()
    out["date"] = out["Start"].dt.date
    out["hour"] = out["Start"].dt.hour
    out["correct"] = out["PlaceError"].eq(0)
    return out


def corner_entropy(visits: pd.DataFrame) -> pd.DataFrame:
    counts = visits.groupby(["AnimalName", "Corner"]).size().rename("n").reset_index()
    counts["p"] = counts["n"] / counts.groupby("AnimalName")["n"].transform("sum")
    entropy = counts.groupby("AnimalName")["p"].apply(lambda p: -(p * np.log2(p)).sum())
    return entropy.rename("corner_entropy_bits").reset_index()


def circadian_metrics(visits: pd.DataFrame, bin_hours: int = 1) -> pd.DataFrame:
    """Compute descriptive IS, IV and RA from hourly visit counts."""
    rows = []
    for animal, frame in visits.groupby("AnimalName"):
        start = frame["Start"].min().floor(f"{bin_hours}h")
        end = frame["Start"].max().ceil(f"{bin_hours}h")
        index = pd.date_range(start, end, freq=f"{bin_hours}h", inclusive="left")
        x = frame.set_index("Start").resample(f"{bin_hours}h").size().reindex(index, fill_value=0).astype(float)
        mean = x.mean()
        denom = ((x - mean) ** 2).sum()
        by_clock = x.groupby(x.index.hour).mean()
        is_value = len(x) * ((by_clock - mean) ** 2).sum() / (24 * denom) if denom else np.nan
        iv_value = len(x) * np.diff(x.to_numpy()).dot(np.diff(x.to_numpy())) / ((len(x) - 1) * denom) if len(x) > 1 and denom else np.nan
        hourly = by_clock.reindex(range(24), fill_value=0).to_numpy()
        m10 = max(np.roll(hourly, -i)[:10].mean() for i in range(24))
        l5 = min(np.roll(hourly, -i)[:5].mean() for i in range(24))
        ra = (m10 - l5) / (m10 + l5) if m10 + l5 else np.nan
        rows.append({"AnimalName": animal, "IS": is_value, "IV": iv_value, "RA": ra, "n_hours": len(x)})
    return pd.DataFrame(rows)

def daily_learning(visits: pd.DataFrame) -> pd.DataFrame:
    x = add_time_fields(visits)
    return x.groupby(["AnimalName", "GroupName", "date"], dropna=False).agg(
        visits=("VisitID", "size"), correct=("correct", "sum"), accuracy=("correct", "mean")
    ).reset_index()


def visit_block_learning(visits: pd.DataFrame, block_size: int = 100) -> pd.DataFrame:
    x = add_time_fields(visits).sort_values(["AnimalName", "Start"])
    x["visit_number"] = x.groupby("AnimalName").cumcount() + 1
    x["block"] = (x["visit_number"] - 1) // block_size + 1
    return x.groupby(["AnimalName", "GroupName", "block"], dropna=False).agg(
        visits=("VisitID", "size"), accuracy=("correct", "mean")
    ).reset_index()


def trials_to_criterion(blocks: pd.DataFrame, threshold: float = 0.5, consecutive: int = 2, block_size: int = 100) -> pd.DataFrame:
    rows = []
    for animal, frame in blocks.sort_values("block").groupby("AnimalName"):
        hit = frame["accuracy"].ge(threshold).rolling(consecutive).sum().eq(consecutive)
        trial = int(frame.loc[hit.idxmax(), "block"] * block_size) if hit.any() else pd.NA
        rows.append({"AnimalName": animal, "trials_to_criterion": trial, "threshold": threshold, "consecutive_blocks": consecutive})
    return pd.DataFrame(rows)
