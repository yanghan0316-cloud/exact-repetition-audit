#!/usr/bin/env python
"""Score true meeting-concatenated cpCER with one assignment per meeting.

Hypotheses are non-overlapping chronological ASR segments attached to a stable
``meeting_stream``/integer ``slot``.  Window manifests and independently
permuted 4 s outputs are deliberately rejected.

Each JSONL row must have exactly this schema (``speaker`` is optional)::

    {"schema":"meeting_stream_hypothesis_segment_v1",
     "session":"R8001_M8004", "split":"eval", "system":"mvdr",
     "asr_model":"model@sha256:<64 lowercase/uppercase hex characters>",
     "meeting_stream":"slot0", "slot":0, "segment_index":0,
     "start":0.0, "end":120.0, "text":"...", "speaker":"opaque_speaker"}

Within a system, every slot must use the same ``meeting_stream`` name across
all meetings.  Within a system/meeting/slot, segment indices are contiguous and
time intervals cannot overlap.  All systems in one report must use one exact
frozen ASR identifier/hash.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from realmeetsep.cpcer import (  # noqa: E402
    ErrorCounts,
    NORMALIZATION_ID,
    StreamTranscript,
    aggregate_counts,
    normalize_zh_text,
    score_cpcer,
    score_sa_cer,
)


REFERENCE_SCHEMA = "meeting_cpcer_reference_v1"
HYPOTHESIS_SCHEMA = "meeting_stream_hypothesis_segment_v1"
ASR_HASH_RE = re.compile(r"^.+@sha256:[0-9a-fA-F]{64}$")
REQUIRED_HYPOTHESIS_FIELDS = {
    "schema",
    "session",
    "split",
    "system",
    "asr_model",
    "meeting_stream",
    "slot",
    "segment_index",
    "start",
    "end",
    "text",
}
ALLOWED_HYPOTHESIS_FIELDS = REQUIRED_HYPOTHESIS_FIELDS | {"speaker"}


def assert_not_test(value: str | Path) -> None:
    parts = [part.casefold() for part in Path(value).parts]
    if any(part in {"test", "test_ali"} or part.startswith("test_ali_") for part in parts):
        raise ValueError(f"Test is sealed; refusing path: {value}")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {line_number} must be a JSON object")
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_reference_rows(rows: list[dict]) -> tuple[dict[str, dict], str]:
    if not rows:
        raise ValueError("No meeting reference records")
    references: dict[str, dict] = {}
    splits: set[str] = set()
    for index, row in enumerate(rows, 1):
        if row.get("schema") != REFERENCE_SCHEMA:
            raise ValueError(
                f"Reference line {index} requires schema={REFERENCE_SCHEMA!r}; "
                "window references are not accepted"
            )
        if row.get("metric_scope") != "full_meeting":
            raise ValueError(f"Reference line {index} is not full_meeting")
        if row.get("character_time_alignment_used") is not False:
            raise ValueError(f"Reference line {index} must declare no character alignment")
        if row.get("normalization") != NORMALIZATION_ID:
            raise ValueError(f"Reference line {index} normalization mismatch")
        split = str(row.get("split", "")).casefold()
        if split not in {"train", "eval"}:
            raise ValueError(f"Reference line {index} split must be train/eval, got {split!r}")
        session = str(row.get("session", ""))
        reference_map = row.get("references")
        speakers = row.get("speakers")
        if not session or not isinstance(reference_map, dict) or not reference_map:
            raise ValueError(f"Reference line {index} requires session and non-empty references")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in reference_map.items()):
            raise ValueError(f"Reference line {index} references must map strings to strings")
        if speakers != list(reference_map):
            raise ValueError(f"Reference line {index} speakers must exactly match reference order")
        if session in references:
            raise ValueError(f"Duplicate reference session: {session}")
        references[session] = row
        splits.add(split)
    if len(splits) != 1:
        raise ValueError(f"One report must contain one split, observed {sorted(splits)}")
    return references, next(iter(splits))


def validate_hypothesis_row(row: dict, index: int) -> None:
    observed = set(row)
    missing = REQUIRED_HYPOTHESIS_FIELDS - observed
    extra = observed - ALLOWED_HYPOTHESIS_FIELDS
    if missing or extra:
        raise ValueError(
            f"Hypothesis line {index} violates {HYPOTHESIS_SCHEMA}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if row["schema"] != HYPOTHESIS_SCHEMA:
        raise ValueError(
            f"Hypothesis line {index} requires schema={HYPOTHESIS_SCHEMA!r}; "
            "window-local hypothesis records are rejected"
        )
    if not isinstance(row["session"], str) or not row["session"]:
        raise ValueError(f"Hypothesis line {index} requires a non-empty session")
    if not isinstance(row["system"], str) or not row["system"]:
        raise ValueError(f"Hypothesis line {index} requires a non-empty system")
    if not isinstance(row["meeting_stream"], str) or not row["meeting_stream"]:
        raise ValueError(f"Hypothesis line {index} requires a stable meeting_stream")
    if isinstance(row["slot"], bool) or not isinstance(row["slot"], int) or row["slot"] < 0:
        raise ValueError(f"Hypothesis line {index} slot must be a non-negative integer")
    if (
        isinstance(row["segment_index"], bool)
        or not isinstance(row["segment_index"], int)
        or row["segment_index"] < 0
    ):
        raise ValueError(f"Hypothesis line {index} segment_index must be non-negative integer")
    if not isinstance(row["text"], str):
        raise ValueError(f"Hypothesis line {index} text must be a string")
    try:
        start = float(row["start"])
        end = float(row["end"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"Hypothesis line {index} start/end must be numeric") from error
    if start < 0.0 or end <= start:
        raise ValueError(f"Hypothesis line {index} requires 0 <= start < end")
    if not ASR_HASH_RE.fullmatch(str(row["asr_model"])):
        raise ValueError(
            f"Hypothesis line {index} asr_model must end with @sha256:<64 hex characters>"
        )
    if row.get("speaker") is not None and not isinstance(row["speaker"], str):
        raise ValueError(f"Hypothesis line {index} speaker must be a string or null")


def build_meeting_streams(
    rows: list[dict], references: dict[str, dict], reference_split: str
) -> tuple[dict[str, dict[str, list[StreamTranscript]]], str]:
    if not rows:
        raise ValueError("No meeting hypothesis records")
    asr_models: set[str] = set()
    systems: dict[str, set[str]] = defaultdict(set)
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    global_stream_for_slot: dict[tuple[str, int], str] = {}
    global_slot_for_stream: dict[tuple[str, str], int] = {}
    record_keys: set[tuple[str, str, int, int]] = set()

    for index, row in enumerate(rows, 1):
        validate_hypothesis_row(row, index)
        split = str(row["split"]).casefold()
        if split != reference_split:
            raise ValueError(
                f"Hypothesis line {index} split mismatch; expected {reference_split}, got {split!r}"
            )
        session = row["session"]
        if session not in references:
            raise ValueError(f"Hypothesis line {index} session absent from references: {session}")
        system = row["system"]
        slot = row["slot"]
        meeting_stream = row["meeting_stream"]
        key = (system, session, slot, row["segment_index"])
        if key in record_keys:
            raise ValueError(f"Duplicate segment key: {key}")
        record_keys.add(key)
        slot_key = (system, slot)
        previous_stream = global_stream_for_slot.setdefault(slot_key, meeting_stream)
        if previous_stream != meeting_stream:
            raise ValueError(
                f"Unstable meeting_stream for system={system}, slot={slot}: "
                f"{previous_stream!r} vs {meeting_stream!r}"
            )
        stream_key = (system, meeting_stream)
        previous_slot = global_slot_for_stream.setdefault(stream_key, slot)
        if previous_slot != slot:
            raise ValueError(
                f"meeting_stream {meeting_stream!r} maps to multiple slots for {system}"
            )
        groups[(system, session, slot)].append(row)
        systems[system].add(session)
        asr_models.add(row["asr_model"])

    if len(asr_models) != 1:
        raise ValueError(
            "One report requires exactly one frozen ASR model/hash across all systems; "
            f"observed {sorted(asr_models)}"
        )
    expected_sessions = set(references)
    for system, observed_sessions in systems.items():
        if observed_sessions != expected_sessions:
            raise ValueError(
                f"System {system} must cover all {len(expected_sessions)} meetings; "
                f"observed {len(observed_sessions)}"
            )

    output: dict[str, dict[str, list[StreamTranscript]]] = defaultdict(dict)
    slots_by_meeting: dict[tuple[str, str], set[int]] = defaultdict(set)
    for (system, session, slot), fragments in sorted(groups.items()):
        ordered = sorted(fragments, key=lambda value: value["segment_index"])
        indices = [row["segment_index"] for row in ordered]
        if indices != list(range(len(ordered))):
            raise ValueError(
                f"Non-contiguous segment_index for system={system}, session={session}, slot={slot}: "
                f"{indices}"
            )
        previous_end = -1.0
        for row in ordered:
            start = float(row["start"])
            if start < previous_end - 1e-9:
                raise ValueError(
                    f"Overlapping/reordered segments for system={system}, session={session}, "
                    f"slot={slot}; meeting cpCER forbids overlapping window concatenation"
                )
            previous_end = float(row["end"])
        speakers = {row.get("speaker") for row in ordered}
        if len(speakers) != 1:
            raise ValueError(
                f"Speaker label changes within system={system}, session={session}, slot={slot}"
            )
        speaker = next(iter(speakers))
        stream = StreamTranscript(
            stream=ordered[0]["meeting_stream"],
            speaker=speaker,
            text="".join(normalize_zh_text(row["text"]) for row in ordered),
        )
        output[system].setdefault(session, []).append(stream)
        slots_by_meeting[(system, session)].add(slot)

    for (system, session), slots in slots_by_meeting.items():
        if slots != set(range(len(slots))):
            raise ValueError(
                f"Slots must be contiguous from zero for system={system}, session={session}: "
                f"{sorted(slots)}"
            )
        # Groups were sorted by slot, but make the relationship explicit.
        output[system][session] = [
            stream
            for _, stream in sorted(
                zip(
                    [global_slot_for_stream[(system, item.stream)] for item in output[system][session]],
                    output[system][session],
                )
            )
        ]
    return output, next(iter(asr_models))


def counts_dict(counts: ErrorCounts) -> dict:
    return counts.to_dict()


def main() -> None:
    args = parse_args()
    for path in (args.references, args.hypotheses, args.output):
        assert_not_test(path)
    reference_rows = read_jsonl(args.references)
    hypothesis_rows = read_jsonl(args.hypotheses)
    references, reference_split = validate_reference_rows(reference_rows)
    streams_by_system, asr_model = build_meeting_streams(
        hypothesis_rows, references, reference_split
    )

    detail_rows: list[dict] = []
    cp_counts_by_system: dict[str, list[ErrorCounts]] = defaultdict(list)
    sa_counts_by_system: dict[str, list[ErrorCounts]] = defaultdict(list)
    for system, meetings in sorted(streams_by_system.items()):
        for session, streams in sorted(meetings.items()):
            reference_map = {
                str(speaker): str(text)
                for speaker, text in references[session]["references"].items()
            }
            cp = score_cpcer(reference_map, streams, already_normalized=True)
            cp_counts_by_system[system].append(cp.counts)
            row = {
                "asr_model": asr_model,
                "cp_assignment": json.dumps(
                    dict(cp.assignment), ensure_ascii=False, sort_keys=True
                ),
                "cp_deletions": cp.counts.deletions,
                "cp_errors": cp.counts.errors,
                "cp_insertions": cp.counts.insertions,
                "cp_substitutions": cp.counts.substitutions,
                "meeting_concatenated_cpcer": cp.counts.cer,
                "reference_chars": cp.counts.reference_chars,
                "sa_available": False,
                "sa_deletions": "",
                "sa_errors": "",
                "sa_insertions": "",
                "sa_substitutions": "",
                "meeting_concatenated_sacer": "",
                "session": session,
                "system": system,
            }
            try:
                sa = score_sa_cer(reference_map, streams, already_normalized=True)
            except ValueError:
                pass
            else:
                sa_counts_by_system[system].append(sa)
                row.update(
                    {
                        "sa_available": True,
                        "sa_deletions": sa.deletions,
                        "sa_errors": sa.errors,
                        "sa_insertions": sa.insertions,
                        "sa_substitutions": sa.substitutions,
                        "meeting_concatenated_sacer": sa.cer,
                    }
                )
            detail_rows.append(row)

    summary: dict = {
        "asr_model": asr_model,
        "audit": {
            "asr_hash_scope": "single_exact_hash_for_entire_report",
            "metric_scope": "full_meeting",
            "overlapping_segments_accepted": False,
            "permutation_scope": "one_assignment_per_system_meeting",
            "report_kind": "meeting_concatenated_cpcer_v1",
            "stable_meeting_stream_slot_required": True,
            "test_used": False,
            "window_local_permutations": False,
        },
        "hypothesis_manifest": str(args.hypotheses.resolve()),
        "normalization": NORMALIZATION_ID,
        "reference_manifest": str(args.references.resolve()),
        "split": reference_split,
        "systems": {},
    }
    for system in sorted(streams_by_system):
        system_rows = [row for row in detail_rows if row["system"] == system]
        cp_values = cp_counts_by_system[system]
        cp_total = aggregate_counts(cp_values)
        sa_values = sa_counts_by_system[system]
        sa_total = aggregate_counts(sa_values)
        summary["systems"][system] = {
            "meetings": len(system_rows),
            "macro_meeting_concatenated_cpcer": float(
                np.mean([value.cer for value in cp_values if value.cer is not None])
            ),
            "micro_meeting_concatenated_cpcer": counts_dict(cp_total),
            "meeting_concatenated_sacer_available_meetings": len(sa_values),
            "meeting_concatenated_sacer_coverage": len(sa_values) / len(system_rows),
            "micro_meeting_concatenated_sacer_on_available": (
                counts_dict(sa_total) if sa_values else None
            ),
            "paper_primary_sacer_valid": len(sa_values) == len(system_rows),
        }
        if sa_values:
            summary["systems"][system]["macro_meeting_concatenated_sacer_on_available"] = float(
                np.mean([value.cer for value in sa_values if value.cer is not None])
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
