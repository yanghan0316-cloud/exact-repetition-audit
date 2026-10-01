"""Reference-free guard for pathological short-period ASR repetitions.

The guard only sees one decoded segment at a time.  Long exact periodic runs
are reduced to a small number of repetitions; ordinary short conversational
repetitions are left untouched by the minimum span and repetition gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RepetitionEvent:
    start: int
    period_chars: int
    repetitions: int
    original_span_chars: int
    kept_span_chars: int
    unit: str


@dataclass(frozen=True)
class GuardResult:
    text: str
    original_characters: int
    guarded_characters: int
    events: tuple[RepetitionEvent, ...]

    @property
    def removed_characters(self) -> int:
        return self.original_characters - self.guarded_characters

    def audit_dict(self) -> dict[str, object]:
        return {
            "events": [asdict(event) for event in self.events],
            "guarded_characters": self.guarded_characters,
            "modified": bool(self.events),
            "original_characters": self.original_characters,
            "removed_characters": self.removed_characters,
        }


def _best_periodic_run(
    text: str,
    *,
    max_period_chars: int,
    min_repetitions: int,
    min_span_chars: int,
    keep_repetitions: int,
) -> tuple[int, int, int] | None:
    """Return ``(start, period, repetitions)`` for the largest removable run."""

    best: tuple[int, int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    length = len(text)
    for start in range(length):
        remaining = length - start
        maximum_period = min(max_period_chars, remaining // min_repetitions)
        for period in range(1, maximum_period + 1):
            unit = text[start : start + period]
            repetitions = 1
            cursor = start + period
            while cursor + period <= length and text[cursor : cursor + period] == unit:
                repetitions += 1
                cursor += period
            span = repetitions * period
            if repetitions < min_repetitions or span < min_span_chars:
                continue
            removable = span - keep_repetitions * period
            # Prefer the greatest reduction, then the shorter primitive period,
            # then the earliest occurrence for deterministic behavior.
            key = (removable, -period, -start)
            if best_key is None or key > best_key:
                best_key = key
                best = (start, period, repetitions)
    return best


def guard_repetitions(
    text: str,
    *,
    max_period_chars: int = 12,
    min_repetitions: int = 6,
    min_span_chars: int = 24,
    keep_repetitions: int = 2,
) -> GuardResult:
    """Reduce pathological exact periodic runs without reference information."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_period_chars <= 0:
        raise ValueError("max_period_chars must be positive")
    if min_repetitions < 3:
        raise ValueError("min_repetitions must be at least three")
    if min_span_chars <= 0:
        raise ValueError("min_span_chars must be positive")
    if keep_repetitions <= 0 or keep_repetitions >= min_repetitions:
        raise ValueError("keep_repetitions must be positive and below min_repetitions")

    original = text
    guarded = text
    events: list[RepetitionEvent] = []
    while True:
        match = _best_periodic_run(
            guarded,
            max_period_chars=max_period_chars,
            min_repetitions=min_repetitions,
            min_span_chars=min_span_chars,
            keep_repetitions=keep_repetitions,
        )
        if match is None:
            break
        start, period, repetitions = match
        unit = guarded[start : start + period]
        original_span = repetitions * period
        kept_span = keep_repetitions * period
        events.append(
            RepetitionEvent(
                start=start,
                period_chars=period,
                repetitions=repetitions,
                original_span_chars=original_span,
                kept_span_chars=kept_span,
                unit=unit,
            )
        )
        guarded = (
            guarded[:start]
            + unit * keep_repetitions
            + guarded[start + original_span :]
        )
    return GuardResult(
        text=guarded,
        original_characters=len(original),
        guarded_characters=len(guarded),
        events=tuple(events),
    )

