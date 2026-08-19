from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_experiment_provenance(output: Path, config: Path, sessions: list[dict]) -> None:
    """Stamp the experiment-level run: which config produced this report.

    Per-session provenance records the raw exports, but without this there is
    nothing tying `report.md` to the `experiment.json` that generated it, so a
    report and a config could drift apart with no way to notice.
    """
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "version": __version__,
               "config": str(Path(config).resolve()), "config_sha256": _sha256(Path(config)),
               "sessions": sessions}
    (output / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_provenance(output: Path, session: Path, parameters: dict) -> None:
    files = [session / "Animals.txt", session / "IntelliCage" / "Visits.txt",
             session / "IntelliCage" / "Nosepokes.txt", session / "IntelliCage" / "HardwareEvents.txt"]
    hashes = {str(path.relative_to(session)): _sha256(path) for path in files if path.exists()}
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "version": __version__,
               "session": str(session), "sha256": hashes, "parameters": parameters}
    (output / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
