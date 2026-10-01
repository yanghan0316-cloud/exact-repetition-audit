#!/usr/bin/env python3
"""Verify the released file bytes against the repository SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checked = 0
    seen: set[str] = set()
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            raise SystemExit(f"Invalid manifest path: {name}")
        seen.add(name)
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            raise SystemExit(f"Missing release file: {name}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise SystemExit(f"SHA-256 mismatch: {name}")
        checked += 1
    if not checked:
        raise SystemExit("Empty release manifest")
    print(json.dumps({"status": "PASS", "files_checked": checked}, sort_keys=True))


if __name__ == "__main__":
    main()
