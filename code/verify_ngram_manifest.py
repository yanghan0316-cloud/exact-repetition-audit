#!/usr/bin/env python3
"""Verify only this addendum's files, including after merging into the base repo."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "MANIFEST_NGRAM.sha256"
checked = 0
for line in manifest.read_text(encoding="utf-8").splitlines():
    expected, relative = line.split("  ", 1)
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Manifest path leaves package: {relative}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"Hash mismatch: {relative}")
    checked += 1
print(json.dumps({"status": "PASS", "addendum_files_verified": checked}))
