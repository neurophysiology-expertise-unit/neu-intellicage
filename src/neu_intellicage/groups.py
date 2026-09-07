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
from math import comb

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
    # Circular quantities such as acrophase cannot go through a difference-of-means
    # test: values straddling midnight average to midday. They are tested by
    # circadian.compare_phase instead and excluded here rather than silently wrong.
    from .circadian import CIRCULAR_MEASURES
    skip = {"AnimalName", "GroupName", "conditioned_visits", *CIRCULAR_MEASURES}
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


CLUSTER_FORMING_THRESHOLD = 2.0
PERMUTATION_ENUMERATION_CAP = 20000


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised difference in means with the small-sample correction.

    The analysis plan requires an effect size with every contrast, significant
    or not. A raw difference in visits per hour is not comparable with a raw
    difference in burstiness; g is, which is what makes a scan across measures
    readable as "which of these is the biggest effect" rather than "which has
    the biggest number". At n=4 per group the correction matters: it shrinks g
    by about 6%.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_df = (len(a) - 1) + (len(b) - 1)
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / pooled_df)
    if pooled_sd == 0:
        return float("nan")
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    return float((a.mean() - b.mean()) / pooled_sd * correction)


def _label_splits(n: int, k: int) -> list[np.ndarray]:
    """Every way of labelling k of n animals as group A."""
    splits = []
    for pick in combinations(range(n), k):
        mask = np.zeros(n, dtype=bool)
        mask[list(pick)] = True
        splits.append(mask)
    return splits


def _find_clusters(statistic: np.ndarray, threshold: float) -> list[dict]:
    """Contiguous same-signed runs of hours above threshold, treating 24h as CIRCULAR.

    Hour 23 is adjacent to hour 0, so a cluster may wrap past midnight. Rotating
    the array to start at a genuine boundary -- an index that does not continue
    the run ending at the previous index -- lets one ordinary linear scan find
    each run exactly once; scanning the raw array would split a wrapping cluster
    in two and report both halves.
    """
    n = len(statistic)
    active = np.abs(statistic) > threshold
    sign = np.sign(statistic)
    if not active.any():
        return []
    if active.all() and len(set(sign)) == 1:
        return [{"hours": list(range(n)), "mass": float(statistic.sum()), "sign": int(sign[0])}]
    rotate_at = next(i for i in range(n)
                     if not (active[i] and active[i - 1] and sign[i] == sign[i - 1]))
    order = [(rotate_at + offset) % n for offset in range(n)]
    runs, start = [], None
    for i in range(n):
        here, previous = order[i], order[i - 1] if i else None
        if active[here] and start is None:
            start = i
        elif active[here] and sign[here] != sign[order[start]]:
            runs.append((start, i - 1)); start = i
        elif not active[here] and start is not None:
            runs.append((start, i - 1)); start = None
    if start is not None:
        runs.append((start, n - 1))
    clusters = []
    for first, last in runs:
        hours = sorted(order[i] for i in range(first, last + 1))
        clusters.append({"hours": hours,
                         "mass": float(sum(statistic[order[i]] for i in range(first, last + 1))),
                         "sign": int(sign[order[first]])})
    return clusters


def cluster_permutation(hourly: pd.DataFrame, groups: dict[str, list[str]],
                        threshold: float = CLUSTER_FORMING_THRESHOLD) -> dict:
    """Where in the day do the groups differ? An hour-by-hour cluster test.

    Collapsing a 24-hour profile into IS, IV or RA answers "is the day shaped
    differently" but never "at which hours", and testing all 24 hours separately
    would be 24 tests. The cluster approach tests contiguous runs of hours as
    single units: hours are standardised against their own null distribution,
    runs exceeding ``threshold`` are formed, and each run's summed statistic is
    compared with the largest run produced by relabelled data. It therefore
    corrects for the 24 comparisons while staying sensitive to an effect spread
    thinly over several adjacent hours -- which is what a phase shift looks like.

    ``hourly`` is one row per animal with columns AnimalName and 0..23. The
    permutation is exact: at four animals per group there are only 70 labellings.
    """
    members = [a for group in groups.values() for a in group]
    frame = hourly.set_index("AnimalName").reindex(members).dropna()
    kept = list(frame.index)
    (name_a, group_a), (name_b, _) = list(groups.items())
    labels = np.array([animal in group_a for animal in kept])
    values = frame[list(range(24))].to_numpy(dtype=float)
    if labels.sum() < 2 or (~labels).sum() < 2:
        return {"clusters": [], "n_animals": len(kept), "n_permutations": 0}

    def difference(mask: np.ndarray) -> np.ndarray:
        return values[mask].mean(axis=0) - values[~mask].mean(axis=0)

    splits = _label_splits(len(kept), int(labels.sum()))
    null = np.array([difference(mask) for mask in splits])
    null_mean, null_sd = null.mean(axis=0), null.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        def standardise(raw: np.ndarray) -> np.ndarray:
            return np.where(null_sd > 0, (raw - null_mean) / null_sd, np.nan)
        observed = standardise(difference(labels))
        null_max = np.array([max((abs(c["mass"]) for c in _find_clusters(standardise(row), threshold)),
                                 default=0.0) for row in null])
    clusters = _find_clusters(observed, threshold)
    for cluster in clusters:
        cluster["p_value"] = float((null_max >= abs(cluster["mass"])).mean())
        cluster["direction"] = f"{name_a} higher" if cluster["sign"] > 0 else f"{name_a} lower"
    return {"clusters": sorted(clusters, key=lambda c: -abs(c["mass"])),
            "n_animals": len(kept), "n_permutations": len(splits),
            "threshold": threshold, "dropped": [a for a in members if a not in kept]}


def permutation_power(n_per_group: int, effect_g: float, trials: int = 2000,
                      alpha: float = 0.05, seed: int = RNG_SEED) -> float:
    """Power of the exact permutation test at a standardised effect ``effect_g``.

    Enumerates the splits once and applies them to every simulated dataset at
    once, because the honest version of this calculation is run over several
    candidate group sizes and the naive loop is minutes of work per size.
    """
    rng = np.random.default_rng(seed)
    total = comb(2 * n_per_group, n_per_group)
    if total <= PERMUTATION_ENUMERATION_CAP:
        masks = np.array([[i in pick for i in range(2 * n_per_group)]
                          for pick in combinations(range(2 * n_per_group), n_per_group)])
    else:
        # Above the cap, enumerating every split costs more than it buys: at
        # n=16 there are 601 million. Sample instead, seeded so the answer is
        # reproducible, and accept that the p-values are Monte Carlo.
        draws = np.argsort(rng.random((PERMUTATION_ENUMERATION_CAP, 2 * n_per_group)), axis=1)
        masks = draws < n_per_group
    data = np.concatenate([rng.normal(effect_g, 1, (trials, n_per_group)),
                           rng.normal(0, 1, (trials, n_per_group))], axis=1)
    differences = np.abs(data @ masks.T - data @ (~masks).T) / n_per_group
    observed = np.abs(data[:, :n_per_group].mean(1) - data[:, n_per_group:].mean(1))[:, None]
    p_values = (differences >= observed - 1e-12).mean(axis=1)
    return float((p_values <= alpha).mean())


def animals_needed(effect_g: float, target_power: float = 0.8,
                   candidates: tuple[int, ...] = (4, 5, 6, 8, 10, 12, 16)) -> dict:
    """Smallest group size reaching ``target_power`` at the observed effect.

    Reported so that a non-significant result carries its own remedy. "Not
    significant at n=4" invites either dropping the measure or running a few
    more animals and hoping; this says which of those is the waste. Note the
    effect size is estimated from the same small sample, so it is itself
    uncertain and this is a planning aid, not a promise.
    """
    if not np.isfinite(effect_g) or effect_g == 0:
        return {"effect_g": effect_g, "needed": None, "power": {}}
    curve = {n: permutation_power(n, abs(effect_g)) for n in candidates}
    needed = next((n for n in candidates if curve[n] >= target_power), None)
    return {"effect_g": abs(effect_g), "needed": needed,
            "target_power": target_power, "power": curve}


def select_headline_measures(scan: pd.DataFrame, profile: pd.DataFrame,
                             max_correlation: float = 0.95,
                             limit: int = 6) -> pd.DataFrame:
    """Thin a scan down to measures that are not restatements of each other.

    A profile scan is not a set of independent tests. Some pairs are dependent
    by construction -- RA is a function of M10 and L5, so on this cohort the
    L5/RA correlation is -1.00 -- and reporting both as separate "hits" triples
    the apparent evidence for one fact. This walks the scan from the strongest
    contrast down and keeps a measure only when its absolute Pearson
    correlation with every already-kept measure is below ``max_correlation``.

    The threshold is deliberately high. With eight animals, any two measures
    that both separate the groups correlate near 1 simply because they encode
    the same eight-way ordering; a threshold of 0.8 would therefore prune
    genuine, distinct findings as "redundant". Only near-deterministic
    dependence -- the kind that comes from one measure being defined in terms of
    another -- is removed here.

    The result is a reading order, NOT a multiplicity correction: the p-values
    are unchanged and the FDR in `scan_profile` is still computed over the full
    scan, because the tests were all performed whether or not they are shown.
    """
    if scan.empty:
        return scan.assign(headline=[], redundant_with=[])
    ranked = scan.sort_values(["p_value", "measure"]).reset_index(drop=True)
    kept, redundant = [], {}
    for measure in ranked["measure"]:
        if measure not in profile:
            continue
        clash = None
        for chosen in kept:
            pair = profile[[measure, chosen]].dropna()
            if len(pair) < 3:
                continue
            r = float(pair[measure].corr(pair[chosen]))
            if np.isfinite(r) and abs(r) >= max_correlation:
                clash = f"{chosen} (r={r:+.2f})"
                break
        if clash is None:
            if len(kept) < limit:
                kept.append(measure)
        else:
            redundant[measure] = clash
    out = ranked.copy()
    out["headline"] = out["measure"].isin(kept)
    out["redundant_with"] = out["measure"].map(redundant).fillna("")
    return out


def _group_arrays(values: pd.DataFrame, measure: str, groups: dict[str, list[str]]):
    (_, members_a), (_, members_b) = list(groups.items())
    series = values.dropna(subset=[measure]).set_index("AnimalName")[measure].astype(float)
    a = series.reindex([m for m in members_a if m in series.index]).to_numpy()
    b = series.reindex([m for m in members_b if m in series.index]).to_numpy()
    return a, b


def session_interaction_p(per_session: pd.DataFrame, measure: str,
                          groups: dict[str, list[str]]) -> dict:
    """Does the group difference in ``measure`` change across sessions?

    ``per_session`` holds one row per animal per session with columns
    ``AnimalName``, ``session`` and ``measure``. The statistic is the spread
    (max minus min) of the per-session group difference; the null is that the
    group LABEL is arbitrary, so the same animal-level permutation is applied to
    every session at once. Permuting each session independently would test a
    different and uninteresting null in which an animal could be knockdown in
    one session and control in the next.

    A non-significant result here is the useful one: it licenses pooling the
    sessions, because it fails to detect a difference that varies between them.
    Note what it cannot do -- with 4 vs 4 this test has the same 2/70 floor and
    far less power than the main contrast, so "no interaction" is weak evidence,
    not a demonstration of homogeneity.
    """
    (name_a, members_a), (name_b, members_b) = list(groups.items())
    frame = per_session.dropna(subset=[measure])
    sessions = sorted(frame["session"].unique())
    animals = [a for a in members_a + members_b
               if a in set(frame["AnimalName"])]
    table = (frame.pivot_table(index="AnimalName", columns="session", values=measure)
             .reindex(index=animals, columns=sessions))
    if table.isna().to_numpy().any() or len(sessions) < 2:
        return {"measure": measure, "sessions": sessions, "p_value": float("nan"),
                "observed_spread": float("nan"), "per_session_difference": {},
                "note": "needs every animal measured in every session, and 2+ sessions"}
    matrix = table.to_numpy(dtype=float)
    n_a = sum(1 for a in animals if a in set(members_a))

    def spread(mask: np.ndarray) -> float:
        differences = matrix[mask].mean(axis=0) - matrix[~mask].mean(axis=0)
        return float(differences.max() - differences.min())

    observed_mask = np.array([a in set(members_a) for a in animals])
    observed = spread(observed_mask)
    extreme = total = 0
    for pick in combinations(range(len(animals)), n_a):
        mask = np.zeros(len(animals), dtype=bool)
        mask[list(pick)] = True
        total += 1
        if spread(mask) >= observed - 1e-12:
            extreme += 1
    per_session = {session: float(matrix[observed_mask, i].mean() - matrix[~observed_mask, i].mean())
                   for i, session in enumerate(sessions)}
    return {"measure": measure, "sessions": sessions, "p_value": extreme / total,
            "observed_spread": observed, "permutations": total,
            "group_a": name_a, "group_b": name_b,
            "per_session_difference": per_session}


def days_to_separation(daily: pd.DataFrame, measure: str, phase: str,
                       groups: dict[str, list[str]]) -> pd.DataFrame:
    """Re-run the contrast on the first k days, for k = 1..n.

    Answers the operational question directly: how long must the cage run before
    the contrast stops flickering? Because the animals are the same every day,
    this curve says nothing about how many ANIMALS are needed -- it is about
    measurement noise per animal, not about the design's resolution floor.
    """
    frame = daily[daily["measure"].eq(measure) & daily["phase"].eq(phase)]
    days = sorted(frame["zt_day"].unique())
    rows = []
    for k in range(1, len(days) + 1):
        window = frame[frame["zt_day"].isin(days[:k])]
        per_animal = (window.groupby("AnimalName", as_index=False)["value"].mean()
                      .rename(columns={"value": measure}))
        try:
            a, b = _group_arrays(per_animal, measure, groups)
            if len(a) < 2 or len(b) < 2:
                continue
            p, _, _ = exact_permutation_p(a, b)
        except (ValueError, KeyError):
            continue
        # Positive margin = a clean gap between the two groups' ranges;
        # negative = the ranges overlap by that much.
        margin = float(max(b.min() - a.max(), a.min() - b.max()))
        rows.append({"days": k, "p_value": p, "difference": float(a.mean() - b.mean()),
                     "hedges_g": hedges_g(a, b), "separated": margin > 0, "margin": margin})
    return pd.DataFrame(rows)
