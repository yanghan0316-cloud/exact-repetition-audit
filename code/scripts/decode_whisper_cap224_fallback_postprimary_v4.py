#!/usr/bin/env python3
"""Decode the frozen 30 s windows with a cap-224 T0 and sampled fallback.

This is a post-primary, fixed-window diagnostic.  It is not OpenAI Whisper's
CLI or long-form transcription loop: windows remain independent, no previous
text is supplied, and no no-speech gate is implemented.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import soundfile as sf
import torch


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
SCRIPTS = PROJECT / "scripts"
for value in (SRC, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import decode_whisper_meeting as frozen  # noqa: E402
from realmeetsep.asr_whisper_confidence_v038 import (  # noqa: E402
    whisper_compression_ratio,
)
from realmeetsep.asr_whisper_fallback_v039 import (  # noqa: E402
    OpenAIAverageLogprob,
    openai_average_logprob_batch_from_scaled_scores,
)
from realmeetsep.meeting_asr import verify_audio_file  # noqa: E402


T0_SCORE_SCHEMA = "whisper_cap224_t0_score_v4"
ATTEMPT_SCHEMA = "whisper_cap224_temperature_fallback_attempt_v4"
SUMMARY_SCHEMA = "whisper_cap224_temperature_fallback_summary_v4"


@dataclass(frozen=True)
class DecodeGroup:
    group_id: str
    group_order: int
    source_manifest_path: str
    source_manifest_sha256: str
    record_indices: tuple[int, ...]


@dataclass(frozen=True)
class DecodeBatch:
    batch_id: str
    global_batch_index: int
    group_id: str
    group_order: int
    within_group_batch_index: int
    record_indices: tuple[int, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def record_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["system"]),
        str(row["session"]),
        int(row["slot"]),
        int(row["segment_index"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("aishell5", "misp2022"), required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--raw-hypotheses", type=Path, required=True)
    parser.add_argument("--decode-batch-map", type=Path, required=True)
    parser.add_argument("--prepare-receipt", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--t0-output", type=Path, required=True)
    parser.add_argument("--t0-scores", type=Path, required=True)
    parser.add_argument("--fallback-output", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def validate_freeze(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    receipt = read_json(args.freeze_receipt)
    if receipt.get("status") != "FROZEN_BEFORE_V4_EXECUTION":
        raise PermissionError("a passing v4 pre-execution freeze receipt is required")
    if receipt.get("protocol_sha256") != sha256(args.protocol):
        raise PermissionError("v4 protocol changed after freeze")
    expected = receipt.get("implementation_sha256", {}).get("decoder")
    if expected != sha256(Path(__file__).resolve()):
        raise PermissionError("v4 decoder changed after freeze")
    expected_core = receipt.get("implementation_sha256", {}).get("fallback_core")
    core = SRC / "realmeetsep" / "asr_whisper_fallback_v039.py"
    if expected_core != sha256(core):
        raise PermissionError("v039 fallback core changed after freeze")
    if (
        protocol.get("status")
        != "FROZEN_BEFORE_POST_PRIMARY_CAP224_CONTROL_EXECUTION"
    ):
        raise PermissionError("v4 protocol status is not frozen")


def validate_inputs(
    audio_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    raw_index: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in raw_rows:
        key = record_key(row)
        if key in raw_index:
            raise ValueError(f"duplicate raw hypothesis key: {key}")
        raw_index[key] = row
    audio_keys = [record_key(row) for row in audio_rows]
    if len(audio_keys) != len(set(audio_keys)):
        raise ValueError("duplicate audio-manifest key")
    if set(audio_keys) != set(raw_index):
        raise ValueError(
            "audio/raw inventories differ: "
            f"audio_only={sorted(set(audio_keys)-set(raw_index))[:3]} "
            f"raw_only={sorted(set(raw_index)-set(audio_keys))[:3]}"
        )
    return raw_index


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_decode_plan(
    audio_rows: Sequence[dict[str, Any]],
    map_rows: Sequence[dict[str, Any]],
    *,
    corpus: str,
    batch_size: int,
    expected_groups: int,
) -> tuple[list[DecodeGroup], list[DecodeBatch], str]:
    """Validate the frozen source-manifest batch map and return its exact plan."""

    if batch_size <= 0:
        raise ValueError("decode-plan batch size must be positive")
    if len(map_rows) != len(audio_rows):
        raise ValueError(
            "decode-batch map/audio cardinality mismatch: "
            f"map={len(map_rows)} audio={len(audio_rows)}"
        )
    audio_keys = [record_key(row) for row in audio_rows]
    map_keys = [record_key(row) for row in map_rows]
    if map_keys != audio_keys:
        if set(map_keys) != set(audio_keys):
            raise ValueError("decode-batch map/audio record inventories differ")
        raise ValueError("decode-batch map does not preserve audio-manifest row order")

    required = {
        "corpus",
        "source_manifest_path",
        "source_manifest_sha256",
        "source_row_index",
        "group_id",
        "group_order",
        "batch_id",
        "batch_position",
        "actual_batch_size",
    }
    for index, row in enumerate(map_rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"decode-batch map row {index} lacks fields: {missing}"
            )
        if row.get("schema") != "whisper_control_decode_batch_map_record_v4":
            raise ValueError(f"decode-batch map row {index} schema differs")
        if str(row["corpus"]) != corpus:
            raise ValueError(f"decode-batch map row {index} corpus differs")
        digest = str(row["source_manifest_sha256"])
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            raise ValueError(
                f"decode-batch map row {index} has invalid manifest SHA-256"
            )
        for field in (
            "source_row_index",
            "group_order",
            "batch_position",
            "actual_batch_size",
        ):
            if isinstance(row[field], bool) or not isinstance(row[field], int):
                raise ValueError(
                    f"decode-batch map row {index} {field} is not an integer"
                )
        if not str(row["source_manifest_path"]):
            raise ValueError(f"decode-batch map row {index} has empty manifest path")
        if not str(row["group_id"]):
            raise ValueError(f"decode-batch map row {index} has empty group_id")
        if not str(row["batch_id"]):
            raise ValueError(f"decode-batch map row {index} has empty batch_id")

    groups: list[DecodeGroup] = []
    group_rows: list[list[int]] = []
    group_meta: dict[str, tuple[int, str, str]] = {}
    current_group_id: str | None = None
    for record_index, row in enumerate(map_rows):
        group_id = str(row["group_id"])
        group_order = int(row["group_order"])
        manifest_path = str(row["source_manifest_path"])
        manifest_sha256 = str(row["source_manifest_sha256"]).lower()
        meta = (group_order, manifest_path, manifest_sha256)
        prior = group_meta.get(group_id)
        if prior is not None and prior != meta:
            raise ValueError(f"decode group metadata changed: {group_id}")
        if group_id != current_group_id:
            if group_id in group_meta:
                raise ValueError(f"decode group is not contiguous: {group_id}")
            if group_order != len(groups):
                raise ValueError(
                    "decode group order is not contiguous: "
                    f"group={group_id} expected={len(groups)} observed={group_order}"
                )
            group_meta[group_id] = meta
            groups.append(
                DecodeGroup(
                    group_id=group_id,
                    group_order=group_order,
                    source_manifest_path=manifest_path,
                    source_manifest_sha256=manifest_sha256,
                    record_indices=(),
                )
            )
            group_rows.append([])
            current_group_id = group_id
        group_rows[-1].append(record_index)

    if len(groups) != expected_groups:
        raise ValueError(
            f"{corpus} decode-group count differs: "
            f"expected={expected_groups} observed={len(groups)}"
        )
    populated_groups: list[DecodeGroup] = []
    batches: list[DecodeBatch] = []
    inventory: list[dict[str, Any]] = []
    aishell_semantic_groups_seen: set[tuple[str, str]] = set()
    for group, indices_list in zip(groups, group_rows, strict=True):
        indices = tuple(indices_list)
        rows = [map_rows[index] for index in indices]
        source_row_indices = [int(row["source_row_index"]) for row in rows]
        if source_row_indices != list(range(len(rows))):
            raise ValueError(
                f"decode group source-row order differs: {group.group_id}"
            )
        if corpus == "aishell5":
            semantic_groups = {
                (str(audio_rows[index]["system"]), str(audio_rows[index]["session"]))
                for index in indices
            }
            if len(semantic_groups) != 1:
                raise ValueError(
                    "AISHELL source manifest crosses (system, session): "
                    f"{group.group_id}"
                )
            semantic_group = next(iter(semantic_groups))
            if semantic_group in aishell_semantic_groups_seen:
                raise ValueError(
                    "AISHELL (system, session) appears in multiple manifests: "
                    f"{semantic_group}"
                )
            aishell_semantic_groups_seen.add(semantic_group)
        populated = DecodeGroup(
            group_id=group.group_id,
            group_order=group.group_order,
            source_manifest_path=group.source_manifest_path,
            source_manifest_sha256=group.source_manifest_sha256,
            record_indices=indices,
        )
        populated_groups.append(populated)

        group_batches: list[DecodeBatch] = []
        cursor = 0
        seen_batch_ids: set[str] = set()
        while cursor < len(rows):
            first = rows[cursor]
            batch_id = str(first["batch_id"])
            actual = int(first["actual_batch_size"])
            if batch_id in seen_batch_ids:
                raise ValueError(
                    f"decode batch is not contiguous: {group.group_id}/{batch_id}"
                )
            if not (1 <= actual <= batch_size):
                raise ValueError(
                    f"decode batch size is invalid: {group.group_id}/{batch_id}"
                )
            stop = cursor + actual
            if stop > len(rows):
                raise ValueError(
                    f"decode batch overruns group: {group.group_id}/{batch_id}"
                )
            selected_rows = rows[cursor:stop]
            if any(str(row["batch_id"]) != batch_id for row in selected_rows):
                raise ValueError(
                    f"decode batch ID changes before actual size: {group.group_id}/{batch_id}"
                )
            if [int(row["batch_position"]) for row in selected_rows] != list(
                range(actual)
            ):
                raise ValueError(
                    f"decode batch positions differ: {group.group_id}/{batch_id}"
                )
            if any(int(row["actual_batch_size"]) != actual for row in selected_rows):
                raise ValueError(
                    f"decode batch size metadata differs: {group.group_id}/{batch_id}"
                )
            if stop < len(rows) and actual != batch_size:
                raise ValueError(
                    f"non-final decode batch is partial: {group.group_id}/{batch_id}"
                )
            batch = DecodeBatch(
                batch_id=batch_id,
                global_batch_index=len(batches),
                group_id=group.group_id,
                group_order=group.group_order,
                within_group_batch_index=len(group_batches),
                record_indices=tuple(indices[cursor:stop]),
            )
            group_batches.append(batch)
            batches.append(batch)
            seen_batch_ids.add(batch_id)
            cursor = stop
        inventory.append(
            {
                "group_id": populated.group_id,
                "group_order": populated.group_order,
                "source_manifest_path": populated.source_manifest_path,
                "source_manifest_sha256": populated.source_manifest_sha256,
                "record_keys": [record_key(audio_rows[index]) for index in indices],
                "batches": [
                    {
                        "batch_id": batch.batch_id,
                        "record_indices": batch.record_indices,
                    }
                    for batch in group_batches
                ],
            }
        )
    flattened = [index for batch in batches for index in batch.record_indices]
    if flattened != list(range(len(audio_rows))):
        raise ValueError("decode batches do not reproduce manifest row order exactly")
    return populated_groups, batches, canonical_json_sha256(inventory)


def validate_decode_group_source_manifests(
    groups: Sequence[DecodeGroup],
    audio_rows: Sequence[dict[str, Any]],
) -> None:
    """Re-hash and replay every map-declared source manifest row-for-row."""

    verified: set[tuple[str, str]] = set()
    for group in groups:
        binding = (group.source_manifest_path, group.source_manifest_sha256)
        if binding in verified:
            raise ValueError(
                "multiple decode groups claim the same source manifest binding: "
                f"{group.source_manifest_path}"
            )
        path = Path(group.source_manifest_path)
        if not path.is_absolute():
            path = PROJECT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != group.source_manifest_sha256:
            raise PermissionError(
                f"decode-group source manifest hash changed: {path}"
            )
        source_rows = read_jsonl(path)
        grouped_audio_rows = [audio_rows[index] for index in group.record_indices]
        if source_rows != grouped_audio_rows:
            source_keys = [record_key(row) for row in source_rows]
            audio_keys = [record_key(row) for row in grouped_audio_rows]
            if source_keys != audio_keys:
                raise PermissionError(
                    "decode-group source manifest record order differs: "
                    f"{path}"
                )
            raise PermissionError(
                "decode-group source manifest row payload differs from the "
                f"prepared audio manifest: {path}"
            )
        verified.add(binding)


def validate_prepare_receipt(
    receipt: dict[str, Any],
    *,
    corpus: str,
    audio_manifest: Path,
    raw_hypotheses: Path,
    decode_batch_map: Path,
    protocol_path: Path,
    freeze_receipt: Path,
    audio_rows: Sequence[dict[str, Any]],
    raw_rows: Sequence[dict[str, Any]],
    map_rows: Sequence[dict[str, Any]],
    groups: Sequence[DecodeGroup],
    batches: Sequence[DecodeBatch],
    partial_batch_count: int,
) -> None:
    if receipt.get("schema") != "whisper_control_predecode_input_binding_receipt_v4":
        raise PermissionError("v4 predecode binding-receipt schema differs")
    if receipt.get("status") != "ALL_CAP224_INPUTS_BOUND_BEFORE_DECODE":
        raise PermissionError("v4 predecode binding receipt is not passing")
    if receipt.get("corpus") != corpus:
        raise PermissionError("v4 predecode binding receipt corpus differs")
    if receipt.get("protocol_sha256") != sha256(protocol_path):
        raise PermissionError("v4 predecode protocol binding changed")
    if receipt.get("freeze_receipt_sha256") != sha256(freeze_receipt):
        raise PermissionError("v4 predecode freeze binding changed")

    expected_files = {
        "all_audio_manifest": (audio_manifest, len(audio_rows)),
        "raw448_hypotheses": (raw_hypotheses, len(raw_rows)),
        "decode_batch_map": (decode_batch_map, len(map_rows)),
    }
    for name, (path, rows) in expected_files.items():
        entry = receipt.get(name)
        if not isinstance(entry, dict):
            raise PermissionError(f"v4 predecode receipt lacks {name}")
        if entry.get("path") != str(path.resolve()):
            raise PermissionError(f"v4 predecode {name} path changed")
        if entry.get("sha256") != sha256(path):
            raise PermissionError(f"v4 predecode {name} hash changed")
        observed_rows = entry.get("rows")
        if (
            isinstance(observed_rows, bool)
            or not isinstance(observed_rows, int)
            or observed_rows != rows
        ):
            raise PermissionError(f"v4 predecode {name} row count changed")
    audio_entry = receipt["all_audio_manifest"]
    if audio_entry.get("every_audio_sha256_verified") is not True:
        raise PermissionError("v4 predecode audio SHA verification is missing")
    map_entry = receipt["decode_batch_map"]
    expected_counts = {
        "groups": len(groups),
        "batches": len(batches),
        "partial_batches": partial_batch_count,
    }
    for name, expected in expected_counts.items():
        observed = map_entry.get(name)
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise PermissionError(
                f"v4 predecode decode-batch-map {name} is not an integer"
            )
        if observed != expected:
            raise PermissionError(
                f"v4 predecode decode-batch-map {name} count changed"
            )


def active_original_decode_batches(
    batches: Sequence[DecodeBatch], active_indices: set[int]
) -> list[tuple[DecodeBatch, tuple[int, ...]]]:
    """Return complete original batches plus active member positions."""

    planned: list[tuple[DecodeBatch, tuple[int, ...]]] = []
    covered: set[int] = set()
    for batch in batches:
        positions = tuple(
            position
            for position, index in enumerate(batch.record_indices)
            if index in active_indices
        )
        if positions:
            planned.append((batch, positions))
            covered.update(batch.record_indices[position] for position in positions)
    if covered != active_indices:
        raise ValueError(
            "active fallback inventory is not covered by original decode batches"
        )
    return planned


def read_waveforms(rows: Sequence[dict[str, Any]]) -> list[Any]:
    arrays: list[Any] = []
    for row in rows:
        path = verify_audio_file(row)
        waveform, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=True
        )
        if sample_rate != 16000 or waveform.shape[1] != 1:
            raise ValueError(f"expected mono 16 kHz audio: {path}")
        expected_samples = int(round(float(row["duration"]) * sample_rate))
        if abs(waveform.shape[0] - expected_samples) > 1:
            raise ValueError(f"manifest/audio duration mismatch: {path}")
        arrays.append(waveform[:, 0])
    return arrays


def expected_prompt_ids(processor: Any, model: Any) -> list[int]:
    prompt_pairs = processor.get_decoder_prompt_ids(
        language="zh", task="transcribe", no_timestamps=True
    )
    positions = [int(position) for position, _ in prompt_pairs]
    if positions != list(range(1, len(prompt_pairs) + 1)):
        raise ValueError("Whisper decoder prompt positions are not contiguous")
    prompt = [int(model.config.decoder_start_token_id)] + [
        int(token_id) for _, token_id in prompt_pairs
    ]
    if len(prompt) != 4:
        raise ValueError(f"expected four-token Mandarin prompt, got {prompt}")
    return prompt


def score_batch(
    generated: Any,
    *,
    temperature: float,
    eos_token_id: int,
    prompt_ids: Sequence[int],
) -> list[OpenAIAverageLogprob]:
    if not generated.scores:
        raise RuntimeError("Whisper generation returned no per-step scores")
    return openai_average_logprob_batch_from_scaled_scores(
        generated.sequences.detach().cpu(),
        generated.scores,
        temperature=temperature,
        eos_token_id=eos_token_id,
        expected_prompt_ids=prompt_ids,
    )


def validate_generated_cardinality(
    generated: Any,
    *,
    expected_batch: int,
    context: str,
    batch_rows: Sequence[dict[str, Any]],
) -> None:
    sequences = getattr(generated, "sequences", None)
    scores = getattr(generated, "scores", None)
    key_digest = hashlib.sha256(
        json.dumps(
            [record_key(row) for row in batch_rows],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise RuntimeError(
            "Whisper fixed-window sequence tensor is not rank 2 before mapping: "
            f"context={context} key_digest={key_digest}"
        )
    observed = int(sequences.shape[0])
    if observed != expected_batch:
        raise RuntimeError(
            "Whisper fixed-window sequence cardinality mismatch before mapping: "
            f"context={context} expected={expected_batch} observed={observed} "
            f"key_digest={key_digest}"
        )
    if not scores:
        raise RuntimeError(
            "Whisper fixed-window generation returned no scores before mapping: "
            f"context={context} key_digest={key_digest}"
        )
    bad_steps = [
        step
        for step, score in enumerate(scores)
        if not isinstance(score, torch.Tensor)
        or score.ndim != 2
        or int(score.shape[0]) != expected_batch
    ]
    if bad_steps:
        raise RuntimeError(
            "Whisper fixed-window score cardinality mismatch before mapping: "
            f"context={context} expected={expected_batch} bad_steps={bad_steps[:8]} "
            f"key_digest={key_digest}"
        )


def decode_and_validate_cardinality(
    processor: Any,
    generated: Any,
    values: Sequence[OpenAIAverageLogprob],
    *,
    expected_batch: int,
    context: str,
    batch_rows: Sequence[dict[str, Any]],
) -> list[str]:
    texts = [
        str(value)
        for value in processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )
    ]
    if len(texts) != expected_batch or len(values) != expected_batch:
        key_digest = hashlib.sha256(
            json.dumps(
                [record_key(row) for row in batch_rows],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        raise RuntimeError(
            "Whisper fixed-window decoded cardinality mismatch before mapping: "
            f"context={context} expected={expected_batch} texts={len(texts)} "
            f"scores={len(values)} key_digest={key_digest}"
        )
    return texts


def timestamp_presence_batch(
    generated: Any,
    *,
    prompt_token_count: int,
    eos_token_id: int,
    timestamp_begin: int,
) -> list[bool]:
    sequences = generated.sequences.detach().cpu()
    flags: list[bool] = []
    for sequence in sequences:
        generated_ids = [int(token) for token in sequence[prompt_token_count:]]
        selected: list[int] = []
        for token in generated_ids:
            selected.append(token)
            if token == eos_token_id:
                break
        flags.append(any(token >= timestamp_begin for token in selected))
    return flags


def generation_common(max_new_tokens: int) -> dict[str, Any]:
    return {
        "language": "zh",
        "task": "transcribe",
        "condition_on_prev_tokens": False,
        "return_timestamps": False,
        "force_unique_generate_call": True,
        "num_beams": 1,
        "num_return_sequences": 1,
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_scores": True,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    for path in (
        args.audio_manifest,
        args.raw_hypotheses,
        args.decode_batch_map,
        args.prepare_receipt,
        args.protocol,
        args.freeze_receipt,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    outputs = (
        args.t0_output,
        args.t0_scores,
        args.fallback_output,
        args.attempts,
        args.summary,
    )
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    protocol = read_json(args.protocol)
    validate_freeze(args, protocol)
    decode = protocol["cap224_decode"]
    fallback = protocol["temperature_fallback"]
    max_new_tokens = int(decode["max_new_tokens"])
    if max_new_tokens != 224:
        raise ValueError("v4 requires max_new_tokens=224")
    if int(fallback["max_new_tokens"]) != max_new_tokens:
        raise ValueError("T0 and fallback max_new_tokens must both equal 224")
    fixed_window_controls = {
        "condition_on_prev_tokens": False,
        "return_timestamps": False,
        "force_unique_generate_call": True,
    }
    for branch_name, branch in (("cap224_decode", decode), ("temperature_fallback", fallback)):
        observed_controls = {
            name: branch.get(name) for name in fixed_window_controls
        }
        if observed_controls != fixed_window_controls:
            raise ValueError(
                f"{branch_name} fixed-window generation controls differ: "
                f"{observed_controls}"
            )
    grouping = protocol.get("decode_grouping")
    if not isinstance(grouping, dict):
        raise ValueError("v4 decode_grouping contract is missing")
    if int(grouping.get("batch_size", -1)) != args.batch_size:
        raise ValueError("decode-grouping batch size differs from runtime")
    corpus_grouping = grouping.get(args.corpus)
    if not isinstance(corpus_grouping, dict):
        raise ValueError(f"v4 decode grouping is missing for {args.corpus}")
    if corpus_grouping.get("within_group_order") != "original_manifest_row_order":
        raise ValueError("decode grouping does not preserve manifest row order")
    if corpus_grouping.get("cross_group_batches") is not False:
        raise ValueError("decode grouping permits cross-group batches")
    if corpus_grouping.get("partial_final_batch_per_group") is not True:
        raise ValueError("decode grouping does not preserve partial final batches")
    expected_groups = int(corpus_grouping["expected_groups"])
    fallback_grouping = grouping.get("fallback")
    expected_fallback_grouping = {
        "grouping": "original_decode_batch_with_inactive_members_retained",
        "batch_size": args.batch_size,
        "adopt_active_records_only": True,
        "attempt_sidecar_active_records_only": True,
        "sampling_seed_batch_index": "original_decode_batch_global_index",
    }
    if fallback_grouping != expected_fallback_grouping:
        raise ValueError("fallback decode-batch grouping differs from v4 contract")
    temperatures = [float(value) for value in fallback["temperatures"]]
    threshold = float(fallback["candidate_acceptance"]["compression_ratio_max"])
    logprob_floor = float(
        fallback["candidate_acceptance"]["avg_logprob_openai_min"]
    )
    protocol_threshold = float(
        protocol["compression_ratio"]["threshold_exclusive"]
    )
    if threshold != 2.4 or protocol_threshold != threshold:
        raise ValueError("all v4 compression thresholds must equal 2.4")
    if logprob_floor != -1.0:
        raise ValueError("v4 OpenAI average-logprob threshold must equal -1.0")
    expected_sampling_controls = {
        "top_k": 0,
        "top_p": 1.0,
        "typical_p": 1.0,
        "min_p": None,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    }
    sampling_controls = dict(fallback["sampling_controls"])
    if sampling_controls != expected_sampling_controls:
        raise ValueError("v4 fallback sampling controls differ from frozen contract")
    if args.batch_size != int(fallback["batch_size"]):
        raise ValueError("batch size differs from frozen v4 protocol")
    if args.device != str(fallback["device"]):
        raise ValueError("device differs from frozen v4 protocol")
    base_seed = int(fallback["sampling_seed"])
    seed_formula = (
        "base_seed + corpus_offset + temperature_index*100000 + "
        "original_decode_batch_global_index"
    )
    if fallback.get("sampling_seed_formula") != seed_formula:
        raise ValueError("v4 fallback sampling-seed formula differs")
    seed_corpus_offsets = fallback.get("sampling_seed_corpus_offsets")
    if seed_corpus_offsets != {"aishell5": 0, "misp2022": 10_000_000}:
        raise ValueError("v4 fallback sampling-seed corpus offsets differ")
    corpus_offset = int(seed_corpus_offsets[args.corpus])

    frozen.verify_frozen_model(args.model_dir.resolve())
    audio_rows = read_jsonl(args.audio_manifest)
    raw_rows = read_jsonl(args.raw_hypotheses)
    batch_map_rows = read_jsonl(args.decode_batch_map)
    raw_index = validate_inputs(audio_rows, raw_rows)
    decode_groups, decode_batches, group_inventory_sha256 = build_decode_plan(
        audio_rows,
        batch_map_rows,
        corpus=args.corpus,
        batch_size=args.batch_size,
        expected_groups=expected_groups,
    )
    validate_decode_group_source_manifests(decode_groups, audio_rows)
    partial_batch_count = sum(
        len(batch.record_indices) < args.batch_size for batch in decode_batches
    )
    prepare_receipt = read_json(args.prepare_receipt)
    validate_prepare_receipt(
        prepare_receipt,
        corpus=args.corpus,
        audio_manifest=args.audio_manifest,
        raw_hypotheses=args.raw_hypotheses,
        decode_batch_map=args.decode_batch_map,
        protocol_path=args.protocol,
        freeze_receipt=args.freeze_receipt,
        audio_rows=audio_rows,
        raw_rows=raw_rows,
        map_rows=batch_map_rows,
        groups=decode_groups,
        batches=decode_batches,
        partial_batch_count=partial_batch_count,
    )
    for row in audio_rows:
        verify_audio_file(row)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(
        str(args.model_dir.resolve()), local_files_only=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model_dir.resolve()), local_files_only=True, dtype=dtype
    ).to(device)
    model.eval()
    prompt_ids = expected_prompt_ids(processor, model)
    eos_token_id = int(model.generation_config.eos_token_id)
    timestamp_begin = int(model.generation_config.no_timestamps_token_id) + 1

    t0_decode_started = time.perf_counter()
    t0_rows: list[dict[str, Any]] = []
    t0_scores: list[dict[str, Any]] = []
    trigger_indices: list[int] = []
    finished_exact = 0
    cap_prefix = 0
    boundary_replacement_rows = 0
    boundary_replacement_characters = 0
    t0_timestamp_token_rows = 0
    processed_t0_records = 0
    for decode_batch in decode_batches:
        batch_indices = list(decode_batch.record_indices)
        batch_rows = [audio_rows[index] for index in batch_indices]
        arrays = read_waveforms(batch_rows)
        processed = processor(
            arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        generated = model.generate(
            processed.input_features.to(device=device, dtype=dtype),
            attention_mask=processed.attention_mask.to(device=device),
            do_sample=False,
            **generation_common(max_new_tokens),
        )
        context = (
            f"{args.corpus}/cap224_t0/group={decode_batch.group_id}/"
            f"batch={decode_batch.batch_id}/global={decode_batch.global_batch_index}"
        )
        validate_generated_cardinality(
            generated,
            expected_batch=len(batch_rows),
            context=context,
            batch_rows=batch_rows,
        )
        values = score_batch(
            generated,
            temperature=0.0,
            eos_token_id=eos_token_id,
            prompt_ids=prompt_ids,
        )
        texts = decode_and_validate_cardinality(
            processor,
            generated,
            values,
            expected_batch=len(batch_rows),
            context=context,
            batch_rows=batch_rows,
        )
        timestamp_flags = timestamp_presence_batch(
            generated,
            prompt_token_count=len(prompt_ids),
            eos_token_id=eos_token_id,
            timestamp_begin=timestamp_begin,
        )
        for local_index, (audio, text, score, has_timestamp_token) in enumerate(
            zip(batch_rows, texts, values, timestamp_flags, strict=True)
        ):
            global_index = batch_indices[local_index]
            raw = raw_index[record_key(audio)]
            raw_text = str(raw["text"])
            decoded = str(text)
            cap_hit = not score.terminated_by_eos
            t0_timestamp_token_rows += int(has_timestamp_token)
            boundary_characters = 0
            prefix_audit_text = decoded
            if score.terminated_by_eos:
                if decoded != raw_text:
                    raise RuntimeError(
                        "cap224 T0 finished-before-cap text mismatch: "
                        f"{record_key(audio)} decoded_sha={hashlib.sha256(decoded.encode()).hexdigest()} "
                        f"raw_sha={hashlib.sha256(raw_text.encode()).hexdigest()}"
                    )
                finished_exact += 1
                prefix_consistent = True
            else:
                if score.generated_step_count != max_new_tokens:
                    raise RuntimeError(
                        "cap224 T0 stopped without EOS before token cap: "
                        f"{record_key(audio)} steps={score.generated_step_count}"
                    )
                prefix_audit_text = decoded.rstrip("\ufffd")
                boundary_characters = len(decoded) - len(prefix_audit_text)
                if boundary_characters:
                    boundary_replacement_rows += 1
                    boundary_replacement_characters += boundary_characters
                prefix_consistent = raw_text.startswith(prefix_audit_text)
                if not prefix_consistent:
                    raise RuntimeError(
                        "cap224 T0 cap-hit text is not a sealed-raw prefix: "
                        f"{record_key(audio)} decoded_sha={hashlib.sha256(decoded.encode()).hexdigest()} "
                        f"raw_sha={hashlib.sha256(raw_text.encode()).hexdigest()}"
                    )
                cap_prefix += 1
            ratio = whisper_compression_ratio(decoded)
            hypothesis = dict(raw)
            hypothesis["text"] = decoded
            t0_rows.append(hypothesis)
            diagnostic = {
                "schema": T0_SCORE_SCHEMA,
                "corpus": args.corpus,
                "system": str(audio["system"]),
                "session": str(audio["session"]),
                "slot": int(audio["slot"]),
                "segment_index": int(audio["segment_index"]),
                "original_decode_group_id": decode_batch.group_id,
                "original_decode_group_order": decode_batch.group_order,
                "original_decode_batch_id": decode_batch.batch_id,
                "original_decode_batch_global_index": decode_batch.global_batch_index,
                "original_decode_batch_position": local_index,
                "original_decode_batch_size": len(batch_rows),
                "compression_ratio": ratio,
                "compression_ratio_gt_2_4": ratio > threshold,
                "cap_hit_without_eos": cap_hit,
                "generated_timestamp_token": bool(has_timestamp_token),
                "prefix_consistent_with_sealed_raw": prefix_consistent,
                "prefix_audit_terminal_u_fffd_characters_stripped": boundary_characters,
                "decoder_boundary_replacement_characters": boundary_characters,
                "prefix_audit_text_sha256": hashlib.sha256(
                    prefix_audit_text.encode("utf-8")
                ).hexdigest(),
                "decoded_text_sha256": hashlib.sha256(
                    decoded.encode("utf-8")
                ).hexdigest(),
                "sealed_raw_text_sha256": hashlib.sha256(
                    raw_text.encode("utf-8")
                ).hexdigest(),
                **score.as_dict(),
            }
            t0_scores.append(diagnostic)
            if ratio > threshold:
                trigger_indices.append(global_index)
        del generated
        processed_t0_records += len(batch_rows)
        print(
            f"{args.corpus} cap224_t0 "
            f"processed={processed_t0_records}/{len(audio_rows)} "
            f"group={decode_batch.group_order + 1}/{len(decode_groups)} "
            f"triggers={len(trigger_indices)}",
            flush=True,
        )

    t0_decode_finished = time.perf_counter()
    active: set[int] = set(trigger_indices)
    selected: dict[int, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    fallback_timestamp_token_attempt_rows = 0
    sampled_member_timestamp_token_rows = 0
    sampled_full_batch_calls = 0
    sampled_member_rows = 0
    active_attempt_rows = 0
    sampled_full_batch_calls_by_temperature: Counter[str] = Counter()
    sampled_member_rows_by_temperature: Counter[str] = Counter()
    active_attempt_rows_by_temperature: Counter[str] = Counter()
    for temperature_index, temperature in enumerate(temperatures):
        if not active:
            break
        temperature_label = f"{temperature:.1f}"
        next_active: set[int] = set()
        planned_batches = active_original_decode_batches(decode_batches, active)
        for call_index, (decode_batch, active_positions) in enumerate(
            planned_batches
        ):
            batch_indices = list(decode_batch.record_indices)
            batch_rows = [audio_rows[index] for index in batch_indices]
            arrays = read_waveforms(batch_rows)
            processed = processor(
                arrays,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            batch_seed = (
                base_seed
                + corpus_offset
                + temperature_index * 100_000
                + decode_batch.global_batch_index
            )
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            generated = model.generate(
                processed.input_features.to(device=device, dtype=dtype),
                attention_mask=processed.attention_mask.to(device=device),
                do_sample=True,
                temperature=temperature,
                **sampling_controls,
                **generation_common(max_new_tokens),
            )
            context = (
                f"{args.corpus}/fallback/T={temperature:.1f}/"
                f"group={decode_batch.group_id}/batch={decode_batch.batch_id}/"
                f"global={decode_batch.global_batch_index}/seed={batch_seed}"
            )
            validate_generated_cardinality(
                generated,
                expected_batch=len(batch_rows),
                context=context,
                batch_rows=batch_rows,
            )
            values = score_batch(
                generated,
                temperature=temperature,
                eos_token_id=eos_token_id,
                prompt_ids=prompt_ids,
            )
            texts = decode_and_validate_cardinality(
                processor,
                generated,
                values,
                expected_batch=len(batch_rows),
                context=context,
                batch_rows=batch_rows,
            )
            timestamp_flags = timestamp_presence_batch(
                generated,
                prompt_token_count=len(prompt_ids),
                eos_token_id=eos_token_id,
                timestamp_begin=timestamp_begin,
            )
            sampled_full_batch_calls += 1
            sampled_member_rows += len(batch_rows)
            sampled_member_timestamp_token_rows += sum(timestamp_flags)
            sampled_full_batch_calls_by_temperature[temperature_label] += 1
            sampled_member_rows_by_temperature[temperature_label] += len(
                batch_rows
            )
            active_attempt_rows += len(active_positions)
            active_attempt_rows_by_temperature[temperature_label] += len(
                active_positions
            )
            for local_index in active_positions:
                audio = batch_rows[local_index]
                text = texts[local_index]
                score = values[local_index]
                has_timestamp_token = timestamp_flags[local_index]
                global_index = batch_indices[local_index]
                decoded = str(text)
                fallback_timestamp_token_attempt_rows += int(
                    has_timestamp_token
                )
                ratio = whisper_compression_ratio(decoded)
                ratio_ok = ratio <= threshold
                logprob_ok = score.avg_logprob_openai >= logprob_floor
                accepted = bool(ratio_ok and logprob_ok)
                final_attempt = temperature_index == len(temperatures) - 1
                raw = raw_index[record_key(audio)]
                attempts.append(
                    {
                        "schema": ATTEMPT_SCHEMA,
                        "corpus": args.corpus,
                        "system": str(audio["system"]),
                        "session": str(audio["session"]),
                        "slot": int(audio["slot"]),
                        "segment_index": int(audio["segment_index"]),
                        "temperature": temperature,
                        "sampling_seed": batch_seed,
                        "original_decode_group_id": decode_batch.group_id,
                        "original_decode_batch_id": decode_batch.batch_id,
                        "original_decode_batch_global_index": (
                            decode_batch.global_batch_index
                        ),
                        "original_decode_batch_position": local_index,
                        "original_decode_batch_size": len(batch_rows),
                        "inactive_original_batch_members_retained": (
                            len(batch_rows) - len(active_positions)
                        ),
                        "generated_timestamp_token": bool(has_timestamp_token),
                        "compression_ratio": ratio,
                        "compression_ratio_pass": ratio_ok,
                        "avg_logprob_openai_pass": logprob_ok,
                        "accepted": accepted,
                        "returned_as_exhausted_last_attempt": bool(
                            final_attempt and not accepted
                        ),
                        "text_sha256": hashlib.sha256(
                            decoded.encode("utf-8")
                        ).hexdigest(),
                        "sealed_raw_text_sha256": hashlib.sha256(
                            str(raw["text"]).encode("utf-8")
                        ).hexdigest(),
                        **score.as_dict(),
                    }
                )
                if accepted or final_attempt:
                    result = dict(t0_rows[global_index])
                    result["text"] = decoded
                    selected[global_index] = result
                else:
                    next_active.add(global_index)
            del generated
            print(
                f"{args.corpus} fallback temperature={temperature:.1f} "
                f"original_batches={call_index + 1}/{len(planned_batches)} "
                f"sampled_members={sampled_member_rows_by_temperature[temperature_label]} "
                f"active_attempts={active_attempt_rows_by_temperature[temperature_label]} "
                f"unresolved_next={len(next_active)}",
                flush=True,
            )
        active = next_active

    if len(selected) != len(trigger_indices):
        raise RuntimeError(
            f"fallback did not return every trigger: "
            f"{len(selected)}/{len(trigger_indices)}"
        )
    if active_attempt_rows != len(attempts):
        raise RuntimeError("active fallback attempt counter differs from sidecar")
    fallback_rows = [dict(row) for row in t0_rows]
    for index, row in selected.items():
        fallback_rows[index] = row

    accepted_distribution = Counter(
        f"{float(row['temperature']):.1f}"
        for row in attempts
        if bool(row["accepted"])
    )
    exhausted = sum(
        bool(row["returned_as_exhausted_last_attempt"]) for row in attempts
    )
    trigger_counts = Counter(
        str(audio_rows[index]["system"]) for index in trigger_indices
    )
    finished = time.perf_counter()
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "CAP224_T0_AND_COMPRESSION_TRIGGERED_FALLBACK_COMPLETE",
        "corpus": args.corpus,
        "elapsed_seconds": finished - started,
        "runtime": {
            "setup_validation_and_model_load_seconds": t0_decode_started - started,
            "cap224_t0_seconds": t0_decode_finished - t0_decode_started,
            "temperature_fallback_seconds": finished - t0_decode_finished,
            "total_seconds": finished - started,
        },
        "input_records": len(audio_rows),
        "cap224_t0": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "condition_on_prev_tokens": False,
            "return_timestamps": False,
            "force_unique_generate_call": True,
            "generated_timestamp_token_rows": t0_timestamp_token_rows,
            "finished_before_cap_exact_match_records": finished_exact,
            "cap_hit_prefix_match_records": cap_prefix,
            "prefix_consistency_failures": 0,
            "terminal_u_fffd_boundary_rows": boundary_replacement_rows,
            "terminal_u_fffd_boundary_characters": boundary_replacement_characters,
            "decoder_boundary_replacement_rows": boundary_replacement_rows,
            "decoder_boundary_replacement_characters": boundary_replacement_characters,
            "prefix_audit_rule": "sealed_raw.startswith(cap224_text.rstrip('\\ufffd')) for cap hits",
            "decode_grouping": {
                "plan_source": "frozen_decode_batch_map",
                "group_count": len(decode_groups),
                "expected_group_count": expected_groups,
                "total_batch_count": len(decode_batches),
                "partial_batch_count": partial_batch_count,
                "batch_size": args.batch_size,
                "cross_group_batches": False,
                "within_group_order": "original_manifest_row_order",
                "group_inventory_sha256": group_inventory_sha256,
                "source_manifest_bindings_rehashed": True,
            },
        },
        "trigger": {
            "rule": "cap224 T0 decoded-text UTF-8 zlib compression ratio > 2.4",
            "records": len(trigger_indices),
            "counts_by_raw_system": dict(sorted(trigger_counts.items())),
            "low_logprob_only_trigger_implemented": False,
        },
        "fallback": {
            "temperatures": temperatures,
            "attempt_records": len(attempts),
            "grouping": "original_decode_batch_with_inactive_members_retained",
            "adopt_active_records_only": True,
            "attempt_sidecar_active_records_only": True,
            "sampling_seed_batch_index": "original_decode_batch_global_index",
            "sampling_seed_formula": seed_formula,
            "sampling_seed_base": base_seed,
            "sampling_seed_corpus_offset": corpus_offset,
            "sampled_full_batch_calls": sampled_full_batch_calls,
            "sampled_member_rows": sampled_member_rows,
            "active_attempt_rows": active_attempt_rows,
            "sampled_member_timestamp_token_rows": sampled_member_timestamp_token_rows,
            "sampled_full_batch_calls_by_temperature": dict(
                sorted(sampled_full_batch_calls_by_temperature.items())
            ),
            "sampled_member_rows_by_temperature": dict(
                sorted(sampled_member_rows_by_temperature.items())
            ),
            "active_attempt_rows_by_temperature": dict(
                sorted(active_attempt_rows_by_temperature.items())
            ),
            "accepted_temperature_distribution": dict(
                sorted(accepted_distribution.items())
            ),
            "exhausted_records": exhausted,
            "single_sample_trajectory_per_temperature": True,
            "condition_on_prev_tokens": False,
            "return_timestamps": False,
            "force_unique_generate_call": True,
            "generated_timestamp_token_attempt_rows": fallback_timestamp_token_attempt_rows,
            "sampling_controls": sampling_controls,
            "best_of": 1,
            "num_beams": 1,
            "num_return_sequences": 1,
        },
        "thresholds": {
            "compression_ratio_max_inclusive": threshold,
            "avg_logprob_openai_min_inclusive": logprob_floor,
        },
        "claim_boundary": {
            "fixed_independent_30_second_windows": True,
            "previous_context": False,
            "return_timestamps": False,
            "force_unique_generate_call": True,
            "batch_cardinality_checked_before_mapping": True,
            "t0_frozen_decode_batch_map_replayed": True,
            "predecode_input_binding_receipt_verified": True,
            "fallback_original_decode_batch_members_retained": True,
            "fallback_attempt_sidecar_active_records_only": True,
            "no_speech_gate": False,
            "official_openai_cli_or_longform_replication": False,
            "openai_avg_logprob_denominator": "sum/(T+1) when no EOS; EOS included otherwise",
        },
        "prompt_ids": prompt_ids,
        "input_audio_manifest": {
            "path": str(args.audio_manifest.resolve()),
            "sha256": sha256(args.audio_manifest),
        },
        "input_prepare_receipt": {
            "path": str(args.prepare_receipt.resolve()),
            "sha256": sha256(args.prepare_receipt),
            "schema": "whisper_control_predecode_input_binding_receipt_v4",
            "status": "ALL_CAP224_INPUTS_BOUND_BEFORE_DECODE",
        },
        "input_decode_batch_map": {
            "path": str(args.decode_batch_map.resolve()),
            "sha256": sha256(args.decode_batch_map),
            "rows": len(batch_map_rows),
            "record_schema": "whisper_control_decode_batch_map_record_v4",
            "group_inventory_sha256": group_inventory_sha256,
            "prepare_receipt_verified": True,
        },
        "input_raw_hypotheses": {
            "path": str(args.raw_hypotheses.resolve()),
            "sha256": sha256(args.raw_hypotheses),
        },
        "outputs": {
            "t0_hypotheses": str(args.t0_output.resolve()),
            "t0_scores": str(args.t0_scores.resolve()),
            "fallback_hypotheses": str(args.fallback_output.resolve()),
            "attempts": str(args.attempts.resolve()),
        },
        "protocol_sha256": sha256(args.protocol),
        "freeze_receipt_sha256": sha256(args.freeze_receipt),
        "decoder_sha256": sha256(Path(__file__).resolve()),
        "fallback_core_sha256": sha256(
            SRC / "realmeetsep" / "asr_whisper_fallback_v039.py"
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    write_jsonl(args.t0_output, t0_rows)
    write_jsonl(args.t0_scores, t0_scores)
    write_jsonl(args.fallback_output, fallback_rows)
    write_jsonl(args.attempts, attempts)
    summary["output_sha256"] = {
        "t0_hypotheses": sha256(args.t0_output),
        "t0_scores": sha256(args.t0_scores),
        "fallback_hypotheses": sha256(args.fallback_output),
        "attempts": sha256(args.attempts),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
