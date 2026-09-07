# neu-intellicage

Reusable Python tooling for IntelliCage session loading, quality control, and
behavioural analysis. The repository contains code only: raw animal-level data
remain outside Git and are supplied at runtime.

The first validation dataset is the Verstreken IntelliCage export from July
2026. It is a test bed, not bundled data and not assumed to be the Tau cohort.

## Install

```bash
python -m pip install -e '.[test]'
```

PyMICE interoperability can be evaluated separately with
`python -m pip install -e '.[pymice]'`. The native loader is intentionally thin
and keeps IntelliCage's protocol-specific `CornerCondition`, `PlaceError`, and
`SideError` columns visible rather than interpreting them silently.

## Commands

```bash
neu-intellicage inventory /path/to/verstreken/Sessions --output outputs/inventory.csv
neu-intellicage qc '/path/to/Sessions/2026-07-13 13.13.43' --output outputs/qc
neu-intellicage tier1 '/path/to/Sessions/2026-07-13 13.13.43' --output outputs/tier1
neu-intellicage tier2 '/path/to/Sessions/2026-07-13 13.13.43' --output outputs/tier2
neu-intellicage all '/path/to/verstreken/Sessions' --session '2026-07-13 13.13.43' --output outputs
neu-intellicage peek '/path/to/Sessions/2026-08-25 13.30.51'          # daily check
neu-intellicage experiment-report experiment.json --output /path/to/project/analysis/experiments/name
scripts/render_report.sh /path/to/project/analysis/experiments/name
```

To create a private, share-ready hourly learning table without RFID tags:

```bash
PYTHONPATH=src python scripts/export_hourly_learning.py \
  '/path/to/one/session' \
  --start '2026-08-14 16:57:11.459' \
  --end '2026-08-17 23:59:49.133' \
  --output '/private/path/hourly_learning.csv' \
  --tau-animals 'Animal 1' 'Animal 2' \
  --scramble-animals 'Animal 5' 'Animal 6'
```

The exporter writes one row per animal and absolute clock hour, retains
zero-visit hours, marks partial hours, and reports visits, licks, conditioned
visits, correct conditioned visits, and visit-based success rate. Its output
and metadata are animal-level study files and must remain outside this code
repository.

To export the matched hourly values behind all-visit and correct-conditioned-
visit actograms, plus hourly success rate per mouse:

```bash
PYTHONPATH=src python scripts/export_actogram_hourly.py \
  '/path/to/one/session' --output '/private/path/actogram_hourly'
```

The canonical CSV stores each date/hour once. Double-plotted figures repeat the
following day in columns 24–47 for readability; those repeated values are not
duplicated in the exported data. Hourly success is correct conditioned visits
divided by all conditioned visits; hours with no conditioned visits are missing,
not zero.

`render_report.sh` builds `report.html` and `report.pdf` from the generated
`report.md` with pandoc, so a delivered PDF can always be rebuilt from the
committed inputs.

The QC command writes visit counts per animal/day and hardware-event counts.
Tier 1 writes hourly activity, inter-visit intervals, corner-use entropy, and
non-parametric circadian IS/IV/RA summaries. Tier 2 writes daily and visit-block
learning curves, terminal accuracy, trials to criterion, error decomposition,
and activity–accuracy data. Figures are PNG files and their plotted values are
also saved as CSV.

## Interpretation safeguards

- Animal identity is keyed by transponder tag and joined to `Animals.txt`.
- Accuracy is defined only over **conditioned** visits (`CornerCondition != 0`).
  IntelliCage sets `PlaceError == 0` both for a correct visit and for every visit
  made while no corner was rewarded, so scoring `PlaceError == 0` alone reports an
  accuracy of 1.000 for habituation and nose-poke sessions. Tables carry
  `conditioned_visits` as the accuracy denominator and leave `accuracy` empty when
  it is zero.
- Terminal accuracy uses each mouse's last **complete** block. A trailing partial
  block of a few visits is reported as such, never scored.
- Trials to criterion counts only complete, adjacent blocks.
- A 25% line is only a geometric four-corner chance reference, not a statistical
  test of learning.
- Group labels are read as recorded. The July validation session labels all four
  animals `Control`; the package does not infer Tau treatment.
- Circadian metrics are descriptive unless the light schedule and complete-day
  recording window are independently confirmed.

## Privacy and provenance

Do not add IntelliCage exports to this repository. Generated tables may contain
transponder tags; review them before sharing. Each analysis command writes a
`provenance.json` file with input path, timestamps, file hashes, package version,
and analysis parameters.

Experiment reports use a JSON configuration to define stable experiment and
session folders. Session stage labels belong in that explicit configuration;
the software does not guess protocol meaning from filenames. The experiment
directory also gets its own `provenance.json` recording the configuration file
and its hash, so a report can always be traced to the configuration that made it.

### Daily peek

`neu-intellicage peek <session>` answers "are they learning yet?" in one screen.
It reads the task from the export -- fixed-target place learning, clockwise
patrolling, or an unconditioned session -- and gives one line per mouse with the
hit rate, the rate that mouse would have to beat given the number of choices it
has actually made, and a verdict. Add `--output DIR` for a cumulative-record
figure and CSVs. Early in a protocol the verdict is usually "not yet decidable",
which is the honest answer and the reason the tool exists.

**Patrolling chance is 1/3, not 1/4.** The rewarded corner is never the one the
animal is standing in, so a mouse that has learned only "do not re-enter the
corner I just left" already scores 1/3. The all-visits rate against 1/4 is
reported alongside, marked as the flattering comparison.

### Circadian analysis

`circadian.py` adds a cosinor fit (mesor, amplitude, acrophase) and the M10/L5
decomposition behind RA, ported from `neu-oldenlabs`. Acrophase is CIRCULAR:
eight mice peaking between 22:56 and 00:36 average to 11:51 by an ordinary mean,
so it is excluded from the linear scan and tested by `compare_phase`, which
permutes a difference of circular means.

`groups.cluster_permutation` tests the 24-hour profile hour by hour, forming
clusters of adjacent hours and comparing each cluster's mass with the largest
produced by relabelled data. It answers "at which hours do the groups differ",
which IS/IV/RA cannot, while correcting for the 24 comparisons. Clock hours are
treated as circular, so a cluster may wrap midnight.

### Between-group statistics

Add `groups` and `group_measures` to `experiment.json` and the report gains a
comparison table computed from code rather than typed into prose. Each measure
yields one value per mouse; the test is an exact two-sided label permutation and
every contrast carries a bootstrap confidence interval on the difference,
whatever the p-value. The table also prints `min_attainable_p`: with four mice
per group the smallest possible p-value is 0.029, so a larger p means the design
could not resolve an effect, not that there is none. Measure kinds are
`nosepoke_probability`, `programmed_target_accuracy`, `daily_accuracy_slope`, and
`preference_shift`; each takes an explicit list of `dates`. Add a `profile_scan`
block to sweep the whole behavioural profile with Benjamini-Hochberg correction;
the report refuses to present a session as a treatment comparison when the groups
did not run under the same corner contingency.

### Activity report (spontaneous behaviour, not learning)

```
neu-intellicage activity-report activity.json --output out/
```

A second, independent report that asks how the mice behave when nothing is being
asked of them. It exists because a spontaneous-activity phenotype does not
require a knockdown to survive a training protocol, and stays measurable when the
contingency changes or a corner breaks — the reasoning `plan.md` takes from
Voikar et al. (2018).

Three conventions differ from the learning report:

- **Zeitgeber time.** A day runs lights-on to lights-on. With a 19:00–07:00 dark
  phase a calendar day splits the night across two rows and puts the halves at
  opposite ends of every plot.
- **Complete days only.** A ZT day is used only when its full 24 h lies inside
  the recorded span. A partial day is not a quiet day: averaging the two together
  deflates every rate and rotates the estimated phase. On the patrolling session,
  partial end days alone produced a spurious −0.2 h/day acrophase drift.
- **Phase splitting.** Every measure that can be computed on a subset of visits is
  computed three times — whole day, light, dark. Measures already defined over the
  24-hour cycle (IS, IV, RA, M10, L5, cosinor) are not split, because a
  light-phase-only IS is a different quantity, not a weaker one.

Sessions are pooled after centring each day on that day's cohort mean, so a
between-session shift in overall activity — the same mice are 58% nocturnal under
free adaptation and 83% during place acquisition — cannot drive a between-group
contrast. Centring subtracts one number from every animal on a day, so each day's
group difference is untouched; only the weighting of days changes.

The report also prints what pooling can and cannot buy. `groups.days_to_separation`
re-runs the contrast on the first k days, which shows whether a measure has
settled; `groups.session_interaction_p` asks whether the difference itself changes
between sessions, permuting the animal label once and applying it to every session
so a mouse cannot change group midway. Neither can lower the design floor: with
four animals per group the smallest attainable p is 2/70 = 0.029 no matter how
many days are recorded, because days are repeated measures on the same animals and
only animals are randomised.

`groups.select_headline_measures` marks measures that are not restatements of one
another (RA is a function of L5, so on this cohort they correlate at −1.00). The
threshold is deliberately high: at n=8 any two measures that both separate the
groups correlate near 1 simply because they encode the same eight-way ordering.
It is a reading order, not a multiplicity correction — the FDR is still computed
over the whole scan.

`exclude_days` drops named ZT days for the whole cohort. Use it for days the cage
was not measuring behaviour -- a corner that stops delivering water changes how
often and how regularly every mouse visits, which no learning measure notices but
every activity measure absorbs. Run the report twice, with and without, and treat
a measure that survives both as the robust one. The shipped example config leaves
it out so the primary run uses every complete day.

Actograms are **single-plotted** by default. Double-plotting keeps a *drifting*
onset continuous across the midnight boundary; under a fixed light schedule with
no drift it prints every datum twice and halves each cell's width. `double: true`
in the config restores the convention. `activity_plots.reward_failure` is the
figure that separates hardware from behaviour: a broken valve fails at every hour
of the day and night, an animal losing interest in a corner does not.
