#!/usr/bin/env python3
"""Create a local-only AISHELL-5 Eval36 locator without reading media bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aishell5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    root = args.aishell5_root.resolve()
    rows = []
    missing = []
    for split_name, role in (
        ("Eval1", "primary"),
        ("Eval2", "exact_replication"),
    ):
        for index in range(1, 19):
            source_session = f"{index:03d}"
            session = f"A5E{1 if split_name == 'Eval1' else 2}_{source_session}"
            directory = root / split_name / source_session
            annotation = directory / "DX01C01.TextGrid"
            channels = [
                directory / f"DX{channel:02d}C01.wav" for channel in range(1, 5)
            ]
            for path in [annotation, *channels]:
                if not path.is_file():
                    missing.append(path.relative_to(root).as_posix())
            rows.append(
                {
                    "schema": "aishell5_eval36_local_locator_v1",
                    "session": session,
                    "analysis_role": role,
                    "source_session": source_session,
                    "annotation_relative_path": annotation.relative_to(root).as_posix(),
                    "channel_relative_paths": [
                        path.relative_to(root).as_posix() for path in channels
                    ],
                }
            )
    if missing and not args.allow_missing:
        raise SystemExit(
            f"missing {len(missing)} expected local files; first={missing[0]}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS" if not missing else "INCOMPLETE_ALLOWED",
                "rows": len(rows),
                "missing": len(missing),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
