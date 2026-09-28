"""Verify the distributed file manifest only; no model or scientific execution."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "docs/submission_manifest.json").read_text(encoding="utf-8"))
    errors = []
    for relative, record in expected.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append(f"Changed: {relative}")
    if (root / ".git").exists():
        print("Local Git history exists; it is outside this manifest's scope.")
    for error in errors:
        print(error)
    print(f"{'FAIL' if errors else 'PASS'}: {len(expected)} distributed file bindings")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
