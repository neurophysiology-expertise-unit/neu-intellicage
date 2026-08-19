"""Between-group comparison with the mouse as the experimental unit.

Every statistic here takes one number per animal and compares two named sets of
animals. The test is an exact two-sided label permutation over all C(n, k)
assignments, which is the honest test at n=4 per group: it makes no
distributional assumption and its resolution limit is visible (with 4 vs 4 the
smallest attainable p is 2/70 = 0.029, so nothing here can ever reach p<0.01).

Following the pre-specified analysis plan, an effect size and a bootstrap
confidence interval are reported for every contrast regardless of significance,
and a non-significant result is reported as absence of evidence, never as
evidence of absence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

RNG_SEED = 20260819


@dataclass(frozen=True)
class Contrast:
    measure: str
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    permutations: int
    min_attainable_p: float

    def as_row(self) -> dict:
        return self.__dict__.copy()


def exact_permutation_p(a: np.ndarray, b: np.ndarray) -> tuple[float, int, float]:
    """Two-sided exact permutation p for the difference in means.

    Enumerates every way of splitting the pooled animals into groups of the
    observed sizes. Returns the p-value, the number of splits enumerated, and
    the smallest p this design could ever produce.
    """
    pooled = np.concatenate([a, b])
    n = len(pooled)
    observed = abs(a.mean() - b.mean())
    indices = range(n)
    extreme = total = 0
    for pick in combinations(indices, len(a)):
        mask = np.zeros(n, dtype=bool)
        mask[list(pick)] = True
        difference = abs(pooled[mask].mean() - pooled[~mask].mean())
        total += 1
        # Ties count as extreme: with small n an exactly-equal split is not
        # evidence against the null and must not be scored in its favour.
        extreme += difference >= observed - 1e-12
    return extreme / total, total, 2.0 / total


def bootstrap_ci(a: np.ndarray, b: np.ndarray, draws: int = 20000,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for the difference in group means.

    At n=4 per group this interval is wide and that width IS the result; it is
    reported so a null is not mistaken for equivalence.
    """
    rng = np.random.default_rng(RNG_SEED)
    differences = np.empty(draws)
    for draw in range(draws):
        differences[draw] = (rng.choice(a, len(a), replace=True).mean()
                             - rng.choice(b, len(b), replace=True).mean())
    tail = (1 - level) / 2
    return float(np.quantile(differences, tail)), float(np.quantile(differences, 1 - tail))


def compare(values: pd.DataFrame, measure: str, groups: dict[str, list[str]]) -> Contrast:
    """Compare one per-animal measure between two named groups.

    ``values`` needs an ``AnimalName`` column and a column named ``measure``
    holding exactly one row per animal.
    """
    (name_a, members_a), (name_b, members_b) = list(groups.items())
    series = values.dropna(subset=[measure]).set_index("AnimalName")[measure].astype(float)
    a = series.reindex([m for m in members_a if m in series.index]).to_numpy()
    b = series.reindex([m for m in members_b if m in series.index]).to_numpy()
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"{measure}: need at least 2 animals per group, got {len(a)} and {len(b)}")
    p, total, floor = exact_permutation_p(a, b)
    low, high = bootstrap_ci(a, b)
    return Contrast(measure=measure, group_a=name_a, group_b=name_b, n_a=len(a), n_b=len(b),
                    mean_a=float(a.mean()), mean_b=float(b.mean()),
                    difference=float(a.mean() - b.mean()), ci_low=low, ci_high=high,
                    p_value=p, permutations=total, min_attainable_p=floor)


def compare_many(values: pd.DataFrame, measures: list[str], groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for measure in measures:
        if measure in values:
            rows.append(compare(values, measure, groups).as_row())
    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR).

    A profile scan tests a dozen or more measures at once, so an uncorrected
    0.03 among fifteen tests is the expected best result under the null, not a
    finding. Note the hard limit this design imposes: with four mice per group
    the smallest raw p is 1/35, so the smallest possible BH value over m tests
    is m/35 -- with fifteen measures NOTHING can be significant after
    correction. The scan is therefore strictly hypothesis-generating, and that
    is a property of n=4, not of the correction.
    """
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.clip(ranked, 0, 1)
    return adjusted


def scan_profile(profile: pd.DataFrame, groups: dict[str, list[str]],
                 measures: list[str] | None = None) -> pd.DataFrame:
    """Run every profile measure through the same contrast, then FDR-correct.

    Measures that are constant across animals carry no information and are
    dropped rather than reported as a null.
    """
    skip = {"AnimalName", "GroupName", "conditioned_visits"}
    candidates = measures or [c for c in profile.columns if c not in skip]
    rows, dropped = [], []
    for measure in candidates:
        values = profile[measure]
        if values.nunique(dropna=True) <= 1 or values.isna().all():
            dropped.append(measure)
            continue
        try:
            rows.append(compare(profile[["AnimalName", measure]], measure, groups).as_row())
        except ValueError:
            dropped.append(measure)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_adjusted_bh"] = benjamini_hochberg(table["p_value"].to_numpy())
    table["dropped_constant_measures"] = ", ".join(dropped) if dropped else ""
    return table.sort_values("p_value").reset_index(drop=True)
