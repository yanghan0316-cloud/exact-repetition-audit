#!/usr/bin/env python3
"""Independent cap-224/n=4 Whisper experiment, retaining original V4 batches.

Pass the full original decode batch map even when the audio manifest contains
only Primary/public. Original global indices are computed before selection.
Atomic complete-batch checkpoints support an exact --resume of the same run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
for directory in (PROJECT / "src", PROJECT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import decode_whisper_meeting as frozen  # noqa: E402
from decode_whisper_cap224_fallback_postprimary_v4 import (  # noqa: E402
    expected_prompt_ids,
    validate_generated_cardinality,
)
from realmeetsep.asr_whisper_confidence_v038 import whisper_compression_ratio  # noqa: E402
from realmeetsep.asr_whisper_ngram_postprimary_v1 import (  # noqa: E402
    constrained_average_logprob_batch,
    validate_token_history,
)
from realmeetsep.meeting_asr import to_hypothesis_row  # noqa: E402

TEMPERATURES = (0.2, 0.4, 0.6, 0.8, 1.0)
BASE_SEED = 20260725
PUBLIC_SYSTEM = "raw_public_sortformer_v2_low_packet_gated"
SCHEMA = "whisper_ngram_blocking_postprimary_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return rows


def record_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return str(row["system"]), str(row["session"]), int(row["slot"]), int(row["segment_index"])


def key_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in ("system", "session", "slot", "segment_index")}


@dataclass(frozen=True)
class OriginalBatch:
    batch_id: str
    global_batch_index: int
    group_id: str
    rows: tuple[dict[str, Any], ...]


def build_original_batches(
    audio_rows: Sequence[dict[str, Any]], map_rows: Sequence[dict[str, Any]],
) -> list[OriginalBatch]:
    """Select complete original batches after deriving indices from the full map."""
    if not map_rows or int(map_rows[0]["group_order"]) != 0 or int(map_rows[0]["source_row_index"]) != 0:
        raise ValueError("pass the complete original batch map, beginning at original group 0")
    audio_by_key = {record_key(row): row for row in audio_rows}
    if len(audio_by_key) != len(audio_rows):
        raise ValueError("duplicate audio-manifest key")
    map_keys = [record_key(row) for row in map_rows]
    if len(map_keys) != len(set(map_keys)):
        raise ValueError("duplicate original batch-map key")
    if not set(audio_by_key).issubset(map_keys):
        raise ValueError("audio records absent from the original batch map")
    if [key for key in map_keys if key in audio_by_key] != list(audio_by_key):
        raise ValueError("audio manifest must preserve the original map record order")
    output: list[OriginalBatch] = []
    seen_ids: set[str] = set()
    cursor = 0
    global_index = 0
    prior_group_order = -1
    group_next_row = 0
    while cursor < len(map_rows):
        first = map_rows[cursor]
        batch_id = str(first["batch_id"])
        actual = int(first["actual_batch_size"])
        group_order = int(first["group_order"])
        if group_order != prior_group_order:
            if group_order != prior_group_order + 1:
                raise ValueError("original batch map has missing/reordered groups")
            prior_group_order = group_order
            group_next_row = 0
        if batch_id in seen_ids or not 1 <= actual <= 8:
            raise ValueError(f"invalid original batch: {batch_id}")
        seen_ids.add(batch_id)
        members = list(map_rows[cursor:cursor + actual])
        if len(members) != actual:
            raise ValueError(f"incomplete original batch: {batch_id}")
        for position, row in enumerate(members):
            if (row["batch_id"] != batch_id or int(row["actual_batch_size"]) != actual
                    or int(row["batch_position"]) != position or row["group_id"] != first["group_id"]
                    or int(row["group_order"]) != group_order
                    or int(row["source_row_index"]) != group_next_row + position
                    or row["corpus"] != "aishell5"):
                raise ValueError(f"original batch positions/group/source order differ: {batch_id}")
            for field in ("original_decode_batch_global_index", "global_batch_index"):
                if field in row and int(row[field]) != global_index:
                    raise ValueError(f"stored original global index disagrees: {batch_id}")
        selected = [record_key(row) in audio_by_key for row in members]
        if any(selected):
            if not all(selected):
                raise ValueError(f"selection splits original batch: {batch_id}")
            rows = tuple(audio_by_key[record_key(row)] for row in members)
            if len({(row["system"], row["session"]) for row in rows}) != 1:
                raise ValueError(f"batch crosses recording/system: {batch_id}")
            output.append(OriginalBatch(batch_id, global_index, str(first["group_id"]), rows))
        cursor += actual
        global_index += 1
        group_next_row += actual
    return output


def generation_parameters(ngram_size: int, temperature: float) -> dict[str, Any]:
    # There is exactly one n source for T0 and every fallback generation.
    result: dict[str, Any] = {
        "language": "zh", "task": "transcribe", "condition_on_prev_tokens": False,
        "return_timestamps": False, "force_unique_generate_call": True,
        "num_beams": 1, "num_return_sequences": 1, "max_new_tokens": 224,
        "return_dict_in_generate": True, "output_scores": True,
        "do_sample": temperature > 0, "no_repeat_ngram_size": ngram_size,
    }
    if temperature > 0:
        result.update({
            "temperature": temperature, "top_k": 0, "top_p": 1.0,
            "typical_p": 1.0, "min_p": None, "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0, "repetition_penalty": 1.0,
        })
    return result


def sampling_seed(temperature_index: int, original_batch_index: int) -> int:
    return BASE_SEED + 100000 * temperature_index + original_batch_index


def validate_protocol(protocol: dict[str, Any]) -> None:
    for name, expected in (("ngram_size", 4), ("max_new_tokens", 224), ("sampling_seed", BASE_SEED)):
        if protocol.get(name) != expected:
            raise ValueError(f"protocol requires {name}={expected}")
    fallback = protocol.get("fallback", {})
    expected_fallback = {
        "initial_temperature": 0.0, "temperatures": list(TEMPERATURES),
        "temperature_index_base": 0, "compression_ratio_max": 2.4,
        "average_logprob_min": -1.0, "accept_first_passing_candidate": True,
        "retain_last_candidate_if_exhausted": True,
    }
    for name, expected in expected_fallback.items():
        if name in fallback and fallback[name] != expected:
            raise ValueError(f"protocol fallback requires {name}={expected}")
    expected_sampling = {key: generation_parameters(4, 0.2)[key] for key in (
        "top_k", "top_p", "typical_p", "min_p", "epsilon_cutoff", "eta_cutoff",
        "repetition_penalty", "no_repeat_ngram_size",
    )}
    if "sampling_controls" in fallback and fallback["sampling_controls"] != expected_sampling:
        raise ValueError("protocol fallback sampling controls differ from fixed experiment")


def audio_info(row: dict[str, Any]) -> Any:
    path = Path(row["audio_filepath"])
    info = sf.info(str(path))
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError(f"expected mono 16 kHz audio: {path}")
    duration = float(row["duration"])
    if not 0 < duration <= 30.0001 or abs(info.frames - round(duration * 16000)) > 1:
        raise ValueError(f"audio/manifest duration mismatch: {path}")
    return info


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--decode-batch-map", type=Path, required=True, help="full V4 map; never pre-filter")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--select-batches", default="", help="comma-separated original global batch indices")
    parser.add_argument("--max-batches", type=int, default=0, help="smoke on first N complete selected original batches")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def decode_batch(
    batch: OriginalBatch, *, processor: Any, model: Any, device: torch.device,
    dtype: torch.dtype, prompt: list[int], ngram_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    arrays = []
    for row in batch.rows:
        audio_info(row)
        array, _ = sf.read(str(row["audio_filepath"]), dtype="float32", always_2d=True)
        arrays.append(array[:, 0])
    processed = processor(arrays, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
    features = processed.input_features.to(device=device, dtype=dtype)
    mask = processed.attention_mask.to(device=device)
    eos = int(model.generation_config.eos_token_id)
    pad = int(model.generation_config.pad_token_id)
    timestamp_begin = int(model.generation_config.no_timestamps_token_id) + 1
    calls: list[dict[str, Any]] = []

    def generate(temperature: float, t_index: int | None, active: set[int]) -> list[dict[str, Any]]:
        seed = None if t_index is None else sampling_seed(t_index, batch.global_batch_index)
        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        parameters = generation_parameters(ngram_size, temperature)
        call_started = time.perf_counter()
        generated = model.generate(features, attention_mask=mask, **parameters)
        validate_generated_cardinality(generated, expected_batch=len(batch.rows), context=batch.batch_id, batch_rows=batch.rows)
        scores = constrained_average_logprob_batch(
            generated.sequences, generated.scores, temperature=temperature,
            eos_token_id=eos, expected_prompt_ids=prompt,
        )
        texts = processor.batch_decode(generated.sequences, skip_special_tokens=True)
        sequences = generated.sequences.detach().cpu().tolist()
        if len(texts) != len(batch.rows) or len(scores) != len(batch.rows):
            raise ValueError("decoded text or score cardinality differs from original batch")
        records: list[dict[str, Any]] = []
        for position, (audio, text, score, sequence) in enumerate(zip(batch.rows, texts, scores, sequences, strict=True)):
            token_audit = validate_token_history(
                sequence, prompt_ids=prompt, score_count=len(generated.scores), eos_token_id=eos,
                pad_token_id=pad, ngram_size=ngram_size, max_new_tokens=224, timestamp_begin=timestamp_begin,
            )
            records.append({
                "schema": SCHEMA, **key_fields(audio),
                "original_decode_batch_id": batch.batch_id,
                "original_decode_batch_global_index": batch.global_batch_index,
                "original_decode_batch_position": position,
                "original_decode_batch_size": len(batch.rows),
                "active_member": position in active,
                "temperature": temperature, "temperature_index": t_index,
                "sampling_seed": seed, "no_repeat_ngram_size": ngram_size,
                "text": str(text), "compression_ratio": whisper_compression_ratio(str(text)),
                **score.as_dict(), **token_audit,
            })
        calls.append({
            "schema": SCHEMA, "original_decode_batch_id": batch.batch_id,
            "original_decode_batch_global_index": batch.global_batch_index,
            "temperature": temperature, "temperature_index": t_index, "sampling_seed": seed,
            "generation_parameters": parameters, "prompt_ids": prompt,
            "score_distribution": "processed constrained scores; temperature removed before normalization",
            "elapsed_seconds": time.perf_counter() - call_started,
            "active_positions": sorted(active), "sampled_batch_size": len(batch.rows),
            "members": records,
        })
        del generated
        return records

    t0 = generate(0.0, None, set(range(len(batch.rows))))
    active = {i for i, row in enumerate(t0) if row["compression_ratio"] > 2.4}
    triggered = set(active)
    final = list(t0)
    attempts: list[dict[str, Any]] = []
    for index, temperature in enumerate(TEMPERATURES):
        if not active:
            break
        candidates = generate(temperature, index, active)
        unresolved: set[int] = set()
        for position in sorted(active):
            candidate = dict(candidates[position])
            ratio_ok = candidate["compression_ratio"] <= 2.4
            logprob_ok = candidate["avg_logprob_openai"] >= -1.0
            accepted = ratio_ok and logprob_ok
            exhausted = index == len(TEMPERATURES) - 1 and not accepted
            candidate.update({
                "compression_ratio_pass": ratio_ok, "avg_logprob_openai_pass": logprob_ok,
                "accepted": accepted, "returned_as_exhausted_last_attempt": exhausted,
                "inactive_original_batch_members_retained": len(batch.rows) - len(active),
            })
            attempts.append(candidate)
            if accepted or exhausted:
                final[position] = candidate
            else:
                unresolved.add(position)
        active = unresolved
    if active:
        raise RuntimeError("fallback left unresolved members")
    t0_scores = [dict(row, triggered_fallback=i in triggered) for i, row in enumerate(t0)]
    diagnostics = []
    for position, row in enumerate(final):
        member_attempts = [item for item in attempts if item["original_decode_batch_position"] == position]
        diagnostics.append({
            "schema": SCHEMA, **key_fields(batch.rows[position]),
            "original_decode_batch_global_index": batch.global_batch_index,
            "original_decode_batch_position": position,
            "final_generated_step_count": row["generated_step_count"],
            "final_temperature": row["temperature"], "cap_hit": row["cap_hit"],
            "terminated_by_eos": row["terminated_by_eos"],
            "fallback_triggered": position in triggered,
            "fallback_accepted": any(item["accepted"] for item in member_attempts),
            "fallback_exhausted": any(item["returned_as_exhausted_last_attempt"] for item in member_attempts),
            "fallback_attempt_count": len(member_attempts),
        })
    return {
        "schema": SCHEMA, "batch_id": batch.batch_id,
        "global_batch_index": batch.global_batch_index,
        "input_rows": list(batch.rows), "ngram_size": ngram_size, "prompt_ids": prompt,
        "t0_hypotheses": [to_hypothesis_row(audio, text=row["text"], asr_model=frozen.ASR_MODEL_ID) for audio, row in zip(batch.rows, t0, strict=True)],
        "fallback_hypotheses": [to_hypothesis_row(audio, text=row["text"], asr_model=frozen.ASR_MODEL_ID) for audio, row in zip(batch.rows, final, strict=True)],
        "t0_scores": t0_scores, "attempts": attempts, "generation_calls": calls,
        "final_diagnostics": diagnostics, "elapsed_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    validate_protocol(protocol)
    ngram_size = int(protocol["ngram_size"])
    audio_rows = read_jsonl(args.audio_manifest)
    map_rows = read_jsonl(args.decode_batch_map)
    batches = build_original_batches(audio_rows, map_rows)
    dimensions = {"expected_records": len(audio_rows), "expected_batches": len(batches), "expected_sessions": len({row["session"] for row in audio_rows})}
    defaults = {"expected_records": 1744, "expected_batches": 222, "expected_sessions": 18}
    for name, observed in dimensions.items():
        if observed != int(protocol.get(name, defaults[name])):
            raise ValueError(f"complete input {name} differs: {observed}")
    if {row["system"] for row in audio_rows} != {PUBLIC_SYSTEM}:
        raise ValueError("this experiment requires public Sortformer input only")
    if any(row.get("analysis_role", "primary") != "primary" for row in audio_rows):
        raise ValueError("non-primary audio row in Primary/public experiment")
    scope = protocol.get("scope", {})
    meeting_manifest = Path(scope.get("meeting_manifest", "manifests/streaming_v035_aishell5_eval36_fullmeeting_v1.jsonl"))
    if not meeting_manifest.is_absolute():
        meeting_manifest = PROJECT / meeting_manifest
    primary_sessions = {row["session"] for row in read_jsonl(meeting_manifest) if row.get("analysis_role") == "primary"}
    if {row["session"] for row in audio_rows} != primary_sessions:
        raise ValueError("audio recording inventory differs from analysis_role=primary in the meeting manifest")
    original_primary_keys = [record_key(row) for row in map_rows if row["system"] == PUBLIC_SYSTEM and row["session"] in primary_sessions]
    if [record_key(row) for row in audio_rows] != original_primary_keys:
        raise ValueError("input omits or reorders original Primary/public windows")
    if args.max_batches < 0:
        raise ValueError("--max-batches must be non-negative")
    if args.select_batches:
        selected = {int(value.strip()) for value in args.select_batches.split(",") if value.strip()}
        existing = {batch.global_batch_index for batch in batches}
        if not selected or not selected.issubset(existing):
            raise ValueError(f"selected original indices absent from input: {sorted(selected - existing)}")
        batches = [batch for batch in batches if batch.global_batch_index in selected]
    if args.max_batches:
        batches = batches[:args.max_batches]
    for batch in batches:
        for row in batch.rows:
            audio_info(row)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint_dir = args.output_dir / "batch_checkpoints"
    metadata_path = args.output_dir / "run_metadata.json"
    if metadata_path.exists() and not args.resume:
        raise FileExistsError("run already exists; use --resume or a new output directory")
    if args.resume and not metadata_path.exists():
        raise FileNotFoundError("--resume requires existing run_metadata.json")
    if not args.resume and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError("checkpoint directory already has data")
    model_audit = frozen.verify_frozen_model(args.model_dir.resolve())
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    if transformers.__version__ != "4.57.6":
        raise RuntimeError(f"requires transformers 4.57.6, observed {transformers.__version__}")
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(str(args.model_dir.resolve()), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(args.model_dir.resolve()), local_files_only=True, dtype=dtype).to(device)
    model.eval()
    prompt = expected_prompt_ids(processor, model)
    model.generation_config.no_repeat_ngram_size = ngram_size
    metadata = {
        "schema": SCHEMA, "protocol": protocol, "model": model_audit,
        "audio_manifest": str(args.audio_manifest.resolve()),
        "decode_batch_map": str(args.decode_batch_map.resolve()),
        "full_input_counts": dimensions,
        "original_global_batch_indices": [batch.global_batch_index for batch in batches],
        "selected_input_rows": [row for batch in batches for row in batch.rows],
        "prompt_ids": prompt, "generation_config": model.generation_config.to_dict(),
        "generation_parameters_by_temperature": {str(t): generation_parameters(ngram_size, t) for t in (0.0, *TEMPERATURES)},
        "seed_formula": "20260725 + 100000 * zero_based_temperature_index + original_global_batch_index",
        "score_interpretation": "constrained processed-score distribution with temperature removed",
        "audio_validation": "readability, mono 16 kHz, duration; prior audio hashes are retained metadata only",
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "soundfile": sf.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "device": str(device), "dtype": str(dtype)},
    }
    if args.resume:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous != metadata:
            raise ValueError("resume configuration, inputs, original batches or environment differ from saved run")
    else:
        atomic_json(metadata_path, metadata)
    completed = []
    resumed = 0
    for number, batch in enumerate(batches, 1):
        checkpoint = checkpoint_dir / f"batch_{batch.global_batch_index:06d}.json"
        if checkpoint.exists():
            value = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (value["batch_id"] != batch.batch_id or value["input_rows"] != list(batch.rows)
                    or value["ngram_size"] != ngram_size or value["prompt_ids"] != prompt
                    or value["global_batch_index"] != batch.global_batch_index):
                raise ValueError(f"checkpoint binding differs: {checkpoint}")
            resumed += 1
        else:
            value = decode_batch(batch, processor=processor, model=model, device=device, dtype=dtype, prompt=prompt, ngram_size=ngram_size)
            atomic_json(checkpoint, value)
        completed.append(value)
        print(f"n={ngram_size} original_batch={batch.global_batch_index} complete={number}/{len(batches)} fallback_attempts={len(value['attempts'])}", flush=True)
    artifacts = {
        "t0_hypotheses": "cap224_ngram_t0_hypotheses.jsonl",
        "fallback_hypotheses": "cap224_ngram_fallback_hypotheses.jsonl",
        "t0_scores": "t0_scores.jsonl", "attempts": "temperature_fallback_attempts.jsonl",
        "generation_calls": "generation_calls.jsonl", "final_diagnostics": "final_decode_diagnostics.jsonl",
    }
    merged = {key: [row for batch in completed for row in batch[key]] for key in artifacts}
    for key, filename in artifacts.items():
        atomic_jsonl(args.output_dir / filename, merged[key])
    diagnostics = merged["final_diagnostics"]
    calls = merged["generation_calls"]
    summary = {
        "schema": SCHEMA, "status": "COMPLETE", "partial_smoke": len(diagnostics) != len(audio_rows),
        "input_records": len(audio_rows), "decoded_records": len(diagnostics), "decoded_batches": len(batches),
        "decoded_sessions": len({row["session"] for row in diagnostics}),
        "original_global_batch_indices": [batch.global_batch_index for batch in batches],
        "ngram_size": ngram_size, "max_new_tokens": 224, "prompt_ids": prompt,
        "trigger_records": sum(row["fallback_triggered"] for row in diagnostics),
        "accepted_records": sum(row["fallback_accepted"] for row in diagnostics),
        "exhausted_records": sum(row["fallback_exhausted"] for row in diagnostics),
        "attempt_records": len(merged["attempts"]),
        "accepted_temperature_distribution": dict(Counter(str(row["temperature"]) for row in merged["attempts"] if row["accepted"])),
        "t0_cap_hit_records": sum(row["cap_hit"] for row in merged["t0_scores"]),
        "final_cap_hit_records": sum(row["cap_hit"] for row in diagnostics),
        "generation_call_cap_hit_member_rows": sum(member["cap_hit"] for call in calls for member in call["members"]),
        "t0_timestamp_token_rows": sum(row["generated_timestamp_token"] for row in merged["t0_scores"]),
        "fallback_timestamp_token_active_attempts": sum(row["generated_timestamp_token"] for row in merged["attempts"]),
        "fallback_timestamp_token_sampled_member_rows": sum(member["generated_timestamp_token"] for call in calls if call["temperature"] > 0 for member in call["members"]),
        "sampled_full_batch_calls": sum(call["temperature"] > 0 for call in calls),
        "sampled_member_rows": sum(call["sampled_batch_size"] for call in calls if call["temperature"] > 0),
        "resumed_batches": resumed,
        "runtime": {"this_invocation_seconds": time.perf_counter() - started,
            "complete_batch_seconds": sum(row["elapsed_seconds"] for row in completed),
            "t0_generation_and_audit_seconds": sum(call["elapsed_seconds"] for call in calls if call["temperature"] == 0),
            "fallback_generation_and_audit_seconds": sum(call["elapsed_seconds"] for call in calls if call["temperature"] > 0)},
        "environment": metadata["environment"],
        "validations": {"complete_original_batches": True, "original_global_indices_preserved": True,
            "fallback_retains_all_original_members": True, "prompt_inclusive_ngram_history": True,
            "finite_scores_through_first_eos": True, "new_path_old_prefix_check_applied": False},
        "outputs": {key: str((args.output_dir / name).resolve()) for key, name in artifacts.items()},
    }
    atomic_json(args.output_dir / "decoder_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
