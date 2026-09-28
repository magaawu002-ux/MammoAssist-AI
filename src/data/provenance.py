"""Small-source provenance; never hash the DICOM collection."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(root: Path, sources: list[tuple[Path, str]],
                     config: dict[str, Any], destination: Path) -> dict:
    """Record original source bytes and the current revision, when available."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "configuration": config,
        "sources": [{"source_filename": p.name,
                     "repository_relative_path": p.relative_to(root).as_posix(),
                     "byte_size": p.stat().st_size, "sha256": sha256_file(p),
                     "source_category": category} for p, category in sources],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
