from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def write_provenance(output: Path, session: Path, parameters: dict) -> None:
    files = [session / "Animals.txt", session / "IntelliCage" / "Visits.txt",
             session / "IntelliCage" / "Nosepokes.txt", session / "IntelliCage" / "HardwareEvents.txt"]
    hashes = {}
    for path in files:
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(path.relative_to(session))] = digest.hexdigest()
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "version": __version__,
               "session": str(session), "sha256": hashes, "parameters": parameters}
    (output / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
