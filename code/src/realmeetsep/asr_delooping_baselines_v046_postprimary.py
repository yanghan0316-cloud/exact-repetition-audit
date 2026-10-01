"""Predeclared post-primary direct baselines for exact ASR delooping.

This module deliberately does not claim to reproduce the delooping procedure
of Barański et al. (ICASSP 2025): their paper defines looping at a high level,
but neither the paper nor the accompanying repository specifies executable
delooping details.  The two transformations here are therefore explicit,
auditable heuristics:

* a left-to-right greedy character-level exact-repeat collapse; and
* an exact-repeat collapse over frozen Whisper content-token IDs.

Both transformations operate on one decoded segment at a time, use no
reference text, retain two repetitions, and return the exact original Python
string when no event is found.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class DeloopingEvent:
    """Content-free trace for one repeat-collapse edit."""

    method: str
    unit_kind: str
    start_unit: int
    period_units: int
    repetitions: int
    original_span_units: int
    kept_span_units: int
    original_start_char: int
    original_span_chars: int
    kept_span_chars: int
    removed_start_char: int
    removed_end_char: int
    removed_characters: int
    unit_sha256: str

    def audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DeloopingResult:
    """Output text and a reversible-by-input content-free edit trace."""

    text: str
    original_characters: int
    output_characters: int
    events: tuple[DeloopingEvent, ...]

    @property
    def removed_characters(self) -> int:
        return self.original_characters - self.output_characters

    def audit_dict(self) -> dict[str, object]:
        return {
            "events": [event.audit_dict() for event in self.events],
            "event_count": len(self.events),
            "modified": bool(self.events),
            "original_characters": self.original_characters,
            "output_characters": self.output_characters,
            "removed_characters": self.removed_characters,
        }


class OffsetTokenizer(Protocol):
    """Minimal interface used from ``WhisperTokenizerFast``."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Any: ...


def _validate_parameters(
    *,
    max_period_units: int,
    min_repetitions: int,
    min_span_units: int,
    keep_repetitions: int,
) -> None:
    if max_period_units <= 0:
        raise ValueError("max_period_units must be positive")
    if min_repetitions < 3:
        raise ValueError("min_repetitions must be at least three")
    if min_span_units <= 0:
        raise ValueError("min_span_units must be positive")
    if keep_repetitions <= 0 or keep_repetitions >= min_repetitions:
        raise ValueError(
            "keep_repetitions must be positive and below min_repetitions"
        )


def _unit_sha256(unit: Sequence[object]) -> str:
    payload = json.dumps(
        list(unit),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _leftmost_character_run(
    text: str,
    *,
    max_period_chars: int,
    min_repetitions: int,
    min_span_chars: int,
) -> tuple[int, int, int] | None:
    """Choose earliest start, then shortest eligible character period."""

    length = len(text)
    for start in range(length):
        remaining = length - start
        maximum_period = min(max_period_chars, remaining // min_repetitions)
        for period in range(1, maximum_period + 1):
            unit = text[start : start + period]
            repetitions = 1
            cursor = start + period
            while (
                cursor + period <= length
                and text[cursor : cursor + period] == unit
            ):
                repetitions += 1
                cursor += period
            if (
                repetitions >= min_repetitions
                and repetitions * period >= min_span_chars
            ):
                return start, period, repetitions
    return None


def collapse_ltr_character_repetitions(
    text: str,
    *,
    max_period_chars: int = 12,
    min_repetitions: int = 6,
    min_span_chars: int = 24,
    keep_repetitions: int = 2,
) -> DeloopingResult:
    """Collapse the leftmost eligible exact character run to two copies.

    After every edit the scan restarts at the beginning.  This makes the
    selection policy simple to inspect and ensures that the returned string is
    a fixed point under the same transformation.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _validate_parameters(
        max_period_units=max_period_chars,
        min_repetitions=min_repetitions,
        min_span_units=min_span_chars,
        keep_repetitions=keep_repetitions,
    )

    original = text
    output = text
    events: list[DeloopingEvent] = []
    while True:
        match = _leftmost_character_run(
            output,
            max_period_chars=max_period_chars,
            min_repetitions=min_repetitions,
            min_span_chars=min_span_chars,
        )
        if match is None:
            break
        start, period, repetitions = match
        unit = output[start : start + period]
        original_span = repetitions * period
        kept_span = keep_repetitions * period
        removed_start = start + kept_span
        removed_end = start + original_span
        events.append(
            DeloopingEvent(
                method="ltr_greedy_character_keep2",
                unit_kind="unicode_code_point",
                start_unit=start,
                period_units=period,
                repetitions=repetitions,
                original_span_units=original_span,
                kept_span_units=kept_span,
                original_start_char=start,
                original_span_chars=original_span,
                kept_span_chars=kept_span,
                removed_start_char=removed_start,
                removed_end_char=removed_end,
                removed_characters=removed_end - removed_start,
                unit_sha256=_unit_sha256(unit),
            )
        )
        output = output[:removed_start] + output[removed_end:]

    return DeloopingResult(
        text=output,
        original_characters=len(original),
        output_characters=len(output),
        events=tuple(events),
    )


def _extract_tokenization(
    tokenizer: OffsetTokenizer, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    try:
        raw_ids = encoded["input_ids"]
        raw_offsets = encoded["offset_mapping"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "tokenizer must return input_ids and offset_mapping"
        ) from error
    if (
        not isinstance(raw_ids, (list, tuple))
        or not isinstance(raw_offsets, (list, tuple))
        or len(raw_ids) != len(raw_offsets)
    ):
        raise ValueError("token IDs and offsets must be equal-length sequences")

    token_ids: list[int] = []
    offsets: list[tuple[int, int]] = []
    previous_start = 0
    for index, (raw_id, raw_offset) in enumerate(
        zip(raw_ids, raw_offsets, strict=True)
    ):
        if (
            isinstance(raw_id, bool)
            or not isinstance(raw_id, int)
            or not isinstance(raw_offset, (list, tuple))
            or len(raw_offset) != 2
        ):
            raise ValueError(f"invalid tokenization item at index {index}")
        start, end = raw_offset
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start <= end <= len(text)
            or start < previous_start
        ):
            raise ValueError(f"invalid token offset at index {index}")
        token_ids.append(int(raw_id))
        offsets.append((start, end))
        previous_start = start
    return token_ids, offsets


def _safe_offset_boundary(
    offsets: Sequence[tuple[int, int]], boundary: int
) -> bool:
    if boundary in {0, len(offsets)}:
        return True
    if not 0 < boundary < len(offsets):
        return False
    # Byte-fallback pieces for one Unicode character can share overlapping
    # offsets.  Never cut between them.
    return offsets[boundary - 1][1] <= offsets[boundary][0]


def _best_token_run(
    token_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    *,
    max_period_tokens: int,
    min_repetitions: int,
    min_span_tokens: int,
    keep_repetitions: int,
) -> tuple[int, int, int] | None:
    """Mirror the v0.35 global tie-break in frozen-token units."""

    best: tuple[int, int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    length = len(token_ids)
    for start in range(length):
        remaining = length - start
        maximum_period = min(max_period_tokens, remaining // min_repetitions)
        for period in range(1, maximum_period + 1):
            unit = token_ids[start : start + period]
            repetitions = 1
            cursor = start + period
            while (
                cursor + period <= length
                and token_ids[cursor : cursor + period] == unit
            ):
                repetitions += 1
                cursor += period
            span = repetitions * period
            if repetitions < min_repetitions or span < min_span_tokens:
                continue
            boundaries = [
                start + repeat_index * period
                for repeat_index in range(repetitions + 1)
            ]
            if not all(
                _safe_offset_boundary(offsets, boundary)
                for boundary in boundaries
            ):
                continue
            removal_start = start + keep_repetitions * period
            removal_end = start + repetitions * period
            if removal_start >= len(offsets):
                continue
            removed_start_char = offsets[removal_start][0]
            removed_end_char = offsets[removal_end - 1][1]
            if removed_end_char <= removed_start_char:
                continue
            removable = span - keep_repetitions * period
            key = (removable, -period, -start)
            if best_key is None or key > best_key:
                best_key = key
                best = start, period, repetitions
    return best


def collapse_whisper_token_repetitions(
    text: str,
    *,
    tokenizer: OffsetTokenizer,
    max_period_tokens: int = 12,
    min_repetitions: int = 6,
    min_span_tokens: int = 24,
    keep_repetitions: int = 2,
) -> DeloopingResult:
    """Collapse exact frozen-Whisper-token runs without whole-text decoding.

    The numeric v0.35 constants are transferred unchanged into token units.
    Offset-safe deletion on the original Python string preserves all text
    outside the removed token occurrences and makes the no-event path exactly
    byte-for-byte identical.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _validate_parameters(
        max_period_units=max_period_tokens,
        min_repetitions=min_repetitions,
        min_span_units=min_span_tokens,
        keep_repetitions=keep_repetitions,
    )

    original = text
    output = text
    events: list[DeloopingEvent] = []
    while True:
        token_ids, offsets = _extract_tokenization(tokenizer, output)
        match = _best_token_run(
            token_ids,
            offsets,
            max_period_tokens=max_period_tokens,
            min_repetitions=min_repetitions,
            min_span_tokens=min_span_tokens,
            keep_repetitions=keep_repetitions,
        )
        if match is None:
            break
        start, period, repetitions = match
        run_end = start + repetitions * period
        kept_end = start + keep_repetitions * period
        original_start_char = offsets[start][0]
        original_end_char = offsets[run_end - 1][1]
        kept_end_char = offsets[kept_end - 1][1]
        removed_start_char = offsets[kept_end][0]
        removed_end_char = original_end_char
        if (
            kept_end_char > removed_start_char
            or removed_end_char <= removed_start_char
        ):
            raise RuntimeError("eligible token run produced an unsafe edit")
        unit = token_ids[start : start + period]
        events.append(
            DeloopingEvent(
                method="whisper_token_exact_keep2",
                unit_kind="whisper_content_token",
                start_unit=start,
                period_units=period,
                repetitions=repetitions,
                original_span_units=repetitions * period,
                kept_span_units=keep_repetitions * period,
                original_start_char=original_start_char,
                original_span_chars=original_end_char - original_start_char,
                kept_span_chars=kept_end_char - original_start_char,
                removed_start_char=removed_start_char,
                removed_end_char=removed_end_char,
                removed_characters=removed_end_char - removed_start_char,
                unit_sha256=_unit_sha256(unit),
            )
        )
        output = output[:removed_start_char] + output[removed_end_char:]

    return DeloopingResult(
        text=output,
        original_characters=len(original),
        output_characters=len(output),
        events=tuple(events),
    )


__all__ = [
    "DeloopingEvent",
    "DeloopingResult",
    "collapse_ltr_character_repetitions",
    "collapse_whisper_token_repetitions",
]
