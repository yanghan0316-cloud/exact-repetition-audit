"""Reference-free helpers for meeting-level separation and ASR export.

This module deliberately works with anonymous FULLTAC slots.  It has no
dependency on TextGrid annotations, close-talk waveforms, target waveforms, or
speaker names.  Those references belong only in the downstream cpCER scorer.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path
import re

import numpy as np
import torch


AUDIO_MANIFEST_SCHEMA = "fullmeeting_asr_audio_segment_v1"
HYPOTHESIS_SCHEMA = "meeting_stream_hypothesis_segment_v1"
SYSTEMS = ("mixture", "estimated_mvdr", "estimated_gated")
SLOT_RE = re.compile(r"^slot([0-9]+)$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def assert_not_test(value: str | Path) -> None:
    """Reject AliMeeting Test paths without rejecting ordinary ``tests`` dirs."""

    parts = [part.casefold() for part in Path(value).parts]
    forbidden = [
        part
        for part in parts
        if part in {"test", "test_ali"}
        or part.startswith("test_ali_")
        or part.endswith("_test_ali")
    ]
    if forbidden:
        raise ValueError(f"AliMeeting Test is sealed; refusing path: {value}")


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a persisted audio artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_audio_file(row: dict) -> Path:
    """Fail closed unless the persisted WAV matches its manifest fingerprint."""

    path = Path(str(row["audio_filepath"]))
    assert_not_test(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest audio is missing: {path}")
    expected_bytes = int(row["audio_bytes"])
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"audio_bytes mismatch for {path}: {observed_bytes} != {expected_bytes}"
        )
    expected_sha256 = str(row["audio_sha256"]).casefold()
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"audio_sha256 mismatch for {path}: {observed_sha256} != {expected_sha256}"
        )
    return path


def slot_index(label: str) -> int:
    match = SLOT_RE.fullmatch(str(label))
    if match is None:
        raise ValueError(f"Expected anonymous slot label slotN, got {label!r}")
    return int(match.group(1))


def valid_profiles_for_session(profiles: dict, session: str) -> list[dict]:
    """Return valid profiles in stable FULLTAC slot order.

    The meeting cpCER schema requires integer slots to be contiguous from zero.
    Renumbering a missing middle slot would silently change anonymous identity,
    so non-prefix profile sets are rejected instead.
    """

    selected = [
        profile
        for key, profile in profiles.items()
        if str(key).startswith(f"{session}/") and int(profile.get("chunks", 0)) > 0
    ]
    selected.sort(key=lambda profile: slot_index(str(profile["slot"])))
    observed = [slot_index(str(profile["slot"])) for profile in selected]
    if observed != list(range(len(observed))):
        raise ValueError(
            f"Valid FULLTAC profiles for {session} are not a zero-based prefix: {observed}"
        )
    if len(selected) < 2:
        raise ValueError(f"At least two valid FULLTAC slots are required for {session}")
    return selected


def aggregate_interference_covariance(
    profiles: Iterable[dict],
    *,
    target_slot: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Combine every other valid anonymous-slot covariance for MVDR.

    ``mean`` is the default and ``sum`` is available for an explicit ablation.
    With trace-relative diagonal loading they produce the same ideal MVDR
    weights up to numerical precision, but recording the choice keeps the
    front-end fully auditable.
    """

    if reduction not in {"mean", "sum"}:
        raise ValueError(f"reduction must be mean or sum, got {reduction!r}")
    candidates: list[torch.Tensor] = []
    for profile in profiles:
        index = slot_index(str(profile["slot"]))
        if index == target_slot:
            continue
        covariance = torch.as_tensor(profile["covariance"])
        if covariance.ndim != 3 or covariance.shape[-1] != covariance.shape[-2]:
            raise ValueError(
                f"Invalid covariance for {profile['slot']}: {tuple(covariance.shape)}"
            )
        candidates.append(covariance)
    if not candidates:
        raise ValueError(f"No interfering covariance remains for target slot{target_slot}")
    shape = candidates[0].shape
    if any(value.shape != shape for value in candidates):
        raise ValueError("Interfering covariance shapes do not match")
    combined = torch.stack(candidates).sum(dim=0)
    if reduction == "mean":
        combined = combined / len(candidates)
    return 0.5 * (combined + combined.transpose(-1, -2).conj())


def posterior_chunk(
    probabilities: np.ndarray,
    *,
    slot: int,
    duration_s: float,
    start_sample: int,
    samples: int,
    sample_rate: int,
) -> torch.Tensor:
    """Interpolate a full-meeting posterior onto exact waveform sample centres."""

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"probabilities must have shape [frames,slots], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    if not 0 <= slot < values.shape[1]:
        raise ValueError(f"slot {slot} outside posterior shape {values.shape}")
    if duration_s <= 0 or start_sample < 0 or samples <= 0 or sample_rate <= 0:
        raise ValueError("invalid posterior interpolation geometry")
    frame_times = (np.arange(values.shape[0], dtype=np.float64) + 0.5) * (
        duration_s / values.shape[0]
    )
    sample_times = (
        start_sample + np.arange(samples, dtype=np.float64) + 0.5
    ) / sample_rate
    track = np.interp(
        sample_times,
        frame_times,
        values[:, slot],
        left=float(values[0, slot]),
        right=float(values[-1, slot]),
    ).astype(np.float32)
    return torch.from_numpy(track).clamp_(0.0, 1.0)


def validate_audio_manifest(rows: list[dict]) -> None:
    """Validate decoder-neutral chunks before frozen-ASR decoding."""

    if not rows:
        raise ValueError("Decoder-neutral audio manifest is empty")
    keys: set[tuple[str, str, int, int]] = set()
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for line_number, row in enumerate(rows, 1):
        if row.get("schema") != AUDIO_MANIFEST_SCHEMA:
            raise ValueError(f"Line {line_number} has unsupported audio manifest schema")
        if str(row.get("split", "")).casefold() != "eval":
            raise ValueError(f"Line {line_number} must have split=eval")
        system = str(row.get("system", ""))
        if system not in SYSTEMS:
            raise ValueError(f"Line {line_number} has unsupported system {system!r}")
        session = str(row.get("session", ""))
        slot = row.get("slot")
        segment = row.get("segment_index")
        if not session or isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise ValueError(f"Line {line_number} has invalid session/slot")
        if isinstance(segment, bool) or not isinstance(segment, int) or segment < 0:
            raise ValueError(f"Line {line_number} has invalid segment_index")
        if row.get("meeting_stream") != f"slot{slot}":
            raise ValueError(f"Line {line_number} has unstable anonymous meeting_stream")
        key = (system, session, slot, segment)
        if key in keys:
            raise ValueError(f"Duplicate audio segment key: {key}")
        keys.add(key)
        start, end = float(row["start"]), float(row["end"])
        if start < 0 or end <= start:
            raise ValueError(f"Line {line_number} requires 0 <= start < end")
        audio_path = Path(str(row["audio_filepath"]))
        assert_not_test(audio_path)
        audio_sha256 = row.get("audio_sha256")
        if not isinstance(audio_sha256, str) or SHA256_RE.fullmatch(audio_sha256) is None:
            raise ValueError(f"Line {line_number} requires audio_sha256 with 64 hex digits")
        audio_bytes = row.get("audio_bytes")
        if (
            isinstance(audio_bytes, bool)
            or not isinstance(audio_bytes, int)
            or audio_bytes <= 0
        ):
            raise ValueError(f"Line {line_number} requires positive integer audio_bytes")
        grouped.setdefault((system, session, slot), []).append(row)
    for group, fragments in grouped.items():
        ordered = sorted(fragments, key=lambda row: row["segment_index"])
        indices = [row["segment_index"] for row in ordered]
        if indices != list(range(len(indices))):
            raise ValueError(f"Non-contiguous segment_index for {group}: {indices}")
        previous_end = -1.0
        for row in ordered:
            start = float(row["start"])
            if start < previous_end - 1e-9:
                raise ValueError(f"Overlapping chunks for {group}")
            previous_end = float(row["end"])


def to_hypothesis_row(audio_row: dict, *, text: str, asr_model: str) -> dict:
    """Project a rich audio record onto the scorer's exact closed schema."""

    return {
        "asr_model": asr_model,
        "end": float(audio_row["end"]),
        "meeting_stream": str(audio_row["meeting_stream"]),
        "schema": HYPOTHESIS_SCHEMA,
        "segment_index": int(audio_row["segment_index"]),
        "session": str(audio_row["session"]),
        "slot": int(audio_row["slot"]),
        "split": "eval",
        "start": float(audio_row["start"]),
        "system": str(audio_row["system"]),
        "text": str(text),
    }

