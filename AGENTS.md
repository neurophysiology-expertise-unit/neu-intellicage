# AGENTS.md — neu-intellicage

- This repository is reusable analysis code, not a study or manuscript repo.
- Never commit raw IntelliCage exports, animal transponder tags, or generated
  animal-level tables. Data are supplied by path at runtime.
- Preserve IntelliCage source semantics. Do not silently reinterpret
  `CornerCondition`, `PlaceError`, `SideError`, or group labels.
- Every plotted quantity must also be written as a machine-readable table.
- Analysis runs must record input hashes and parameters in `provenance.json`.
- Treat the Verstreken July 2026 dataset as validation data only. Do not infer
  that it is the Tau cohort unless its owner confirms this independently.
