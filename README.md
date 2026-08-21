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

### Between-group statistics

Add `groups` and `group_measures` to `experiment.json` and the report gains a
comparison table computed from code rather than typed into prose. Each measure
yields one value per mouse; the test is an exact two-sided label permutation and
every contrast carries a bootstrap confidence interval on the difference,
whatever the p-value. The table also prints `min_attainable_p`: with four mice
per group the smallest possible p-value is 0.029, so a larger p means the design
could not resolve an effect, not that there is none. Measure kinds are
`nosepoke_probability`, `programmed_target_accuracy`, `daily_accuracy_slope`, and
`preference_shift`; each takes an explicit list of `dates`.
