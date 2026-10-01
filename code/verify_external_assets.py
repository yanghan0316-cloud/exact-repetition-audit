#!/usr/bin/env python3
"""Print pinned external assets and verify optional authorized local files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ASSETS = {
    "sortformer": {
        "model_id": "nvidia/diar_streaming_sortformer_4spk-v2",
        "revision": "59620264e008f2b06a0e969688e0af3e8705478b",
        "sha256": "b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329",
        "license": "CC-BY-4.0",
    },
    "whisper": {
        "model_id": "openai/whisper-base",
        "revision": "e37978b90ca9030d5170a5c07aadb050351a65bb",
        "sha256": "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b",
        "license_metadata": "Apache-2.0",
    },
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sortformer-archive", type=Path)
    parser.add_argument("--whisper-weights", type=Path)
    args = parser.parse_args()
    checks = {}
    for name, path in (
        ("sortformer", args.sortformer_archive),
        ("whisper", args.whisper_weights),
    ):
        if path is None:
            checks[name] = "NOT_CHECKED"
            continue
        observed = digest(path)
        expected = ASSETS[name]["sha256"]
        checks[name] = {
            "status": "PASS" if observed == expected else "FAIL",
            "observed_sha256": observed,
        }
        if observed != expected:
            raise SystemExit(
                json.dumps({"assets": ASSETS, "checks": checks}, indent=2, sort_keys=True)
            )
    print(json.dumps({"assets": ASSETS, "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
