#!/usr/bin/env python
"""Decode meeting audio chunks with one frozen, strictly local Whisper Base.

Every system is decoded with the identical model bytes and deterministic
Mandarin transcription settings.  Network fallback is disabled before
Transformers is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import soundfile as sf
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from realmeetsep.meeting_asr import (  # noqa: E402
    assert_not_test,
    to_hypothesis_row,
    validate_audio_manifest,
    verify_audio_file,
)


WHISPER_REPOSITORY = "openai/whisper-base"
WHISPER_COMMIT = "e37978b90ca9030d5170a5c07aadb050351a65bb"
WHISPER_WEIGHTS_SHA256 = "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b"
WHISPER_CONFIG_SHA256 = "a153c53883a6799b6f056b4a8d1a515c9926d03994682ba88a7616618d7da0c1"
ASR_MODEL_ID = f"{WHISPER_REPOSITORY}@sha256:{WHISPER_WEIGHTS_SHA256}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_model(model_dir: Path) -> dict:
    assert_not_test(model_dir)
    weights = model_dir / "model.safetensors"
    config = model_dir / "config.json"
    metadata = (
        model_dir
        / ".cache"
        / "huggingface"
        / "download"
        / "model.safetensors.metadata"
    )
    for path in (weights, config, metadata):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Whisper artifact missing: {path}")
    weights_hash = sha256(weights)
    config_hash = sha256(config)
    if weights_hash != WHISPER_WEIGHTS_SHA256:
        raise ValueError(
            f"Whisper weights hash mismatch: {weights_hash} != {WHISPER_WEIGHTS_SHA256}"
        )
    if config_hash != WHISPER_CONFIG_SHA256:
        raise ValueError(
            f"Whisper config hash mismatch: {config_hash} != {WHISPER_CONFIG_SHA256}"
        )
    metadata_lines = metadata.read_text(encoding="utf-8").splitlines()
    if not metadata_lines or metadata_lines[0].strip() != WHISPER_COMMIT:
        raise ValueError(
            f"Whisper Hub commit mismatch: expected {WHISPER_COMMIT}, got "
            f"{metadata_lines[0].strip() if metadata_lines else '<empty>'}"
        )
    return {
        "asr_model": ASR_MODEL_ID,
        "commit": WHISPER_COMMIT,
        "config_sha256": config_hash,
        "repository": WHISPER_REPOSITORY,
        "weights_sha256": weights_hash,
    }


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Audio manifest line {line_number} must be an object")
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models" / "whisper-base")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    for path in (args.audio_manifest, args.model_dir, args.output):
        assert_not_test(path.resolve())
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite hypothesis file: {args.output}")
    if args.batch_size <= 0 or args.max_records < 0:
        raise ValueError("--batch-size must be positive and --max-records non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    model_audit = verify_frozen_model(args.model_dir.resolve())
    source_rows = read_jsonl(args.audio_manifest)
    validate_audio_manifest(source_rows)
    rows = source_rows[: args.max_records] if args.max_records else source_rows

    # These are set before importing Transformers/Hugging Face Hub.  Any
    # missing local artifact must fail instead of silently reaching the web.
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
        str(args.model_dir.resolve()),
        local_files_only=True,
        dtype=dtype,
    ).to(device)
    model.eval()

    hypothesis_rows: list[dict] = []
    for offset in range(0, len(rows), args.batch_size):
        batch_rows = rows[offset : offset + args.batch_size]
        arrays = []
        for row in batch_rows:
            audio_path = verify_audio_file(row)
            waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
            if sample_rate != 16000 or waveform.shape[1] != 1:
                raise ValueError(
                    f"Whisper input must be mono 16 kHz, got {waveform.shape}/{sample_rate}: "
                    f"{audio_path}"
                )
            expected = int(round(float(row["duration"]) * sample_rate))
            if abs(waveform.shape[0] - expected) > 1:
                raise ValueError(
                    f"Audio/manifest duration mismatch: {audio_path}, "
                    f"frames={waveform.shape[0]}, expected={expected}"
                )
            arrays.append(waveform[:, 0])
        processed = processor(
            arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        features = processed.input_features.to(device=device, dtype=dtype)
        attention_mask = processed.attention_mask.to(device=device)
        generated = model.generate(
            features,
            attention_mask=attention_mask,
            language="zh",
            task="transcribe",
            do_sample=False,
            num_beams=1,
        )
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        for row, decoded in zip(batch_rows, texts, strict=True):
            hypothesis_rows.append(
                to_hypothesis_row(row, text=decoded, asr_model=ASR_MODEL_ID)
            )
        print(f"Whisper decoded {min(offset + len(batch_rows), len(rows))}/{len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in hypothesis_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "asr_model": ASR_MODEL_ID,
        "audio_hashes_verified": True,
        "audit": {
            "all_systems_same_model": True,
            "audio_hashes_verified": True,
            "local_files_only": True,
            "network_fallback": False,
            "test_used": False,
        },
        "decoded_records": len(hypothesis_rows),
        "device": str(device),
        "elapsed_s": time.perf_counter() - started,
        "generation": {
            "do_sample": False,
            "language": "zh",
            "num_beams": 1,
            "task": "transcribe",
        },
        "input_manifest": str(args.audio_manifest.resolve()),
        "input_manifest_sha256": sha256(args.audio_manifest),
        "input_records": len(source_rows),
        "model": model_audit,
        "output": str(args.output.resolve()),
        "output_hypotheses_sha256": sha256(args.output),
        "partial_smoke": bool(args.max_records and args.max_records < len(source_rows)),
        "systems": sorted({row["system"] for row in hypothesis_rows}),
        "verified_audio_records": len(rows),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()



