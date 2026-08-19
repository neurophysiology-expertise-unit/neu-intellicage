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
```

The QC command writes visit counts per animal/day and hardware-event counts.
Tier 1 writes hourly activity, inter-visit intervals, corner-use entropy, and
non-parametric circadian IS/IV/RA summaries. Tier 2 writes daily and visit-block
learning curves, terminal accuracy, trials to criterion, error decomposition,
and activity–accuracy data. Figures are PNG files and their plotted values are
also saved as CSV.

## Interpretation safeguards

- Animal identity is keyed by transponder tag and joined to `Animals.txt`.
- `PlaceError == 0` is treated as a correct-place visit; this is reported
  explicitly in outputs.
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
the software does not guess protocol meaning from filenames.
