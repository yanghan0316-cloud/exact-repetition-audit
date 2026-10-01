"""Reference-free confidence summaries for deterministic Whisper generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import zlib
from typing import Iterable, Sequence


@dataclass(frozen=True)
class WhisperConfidence:
    """Compact metrics computed from one already-generated Whisper sequence."""

    compression_ratio: float
    content_token_count: int
    decoded_text_characters: int
    decoded_text_utf8_bytes: int
    eos_logprob: float | None
    generated_step_count: int
    max_length_reached: bool
    mean_generated_token_logprob: float | None
    mean_generated_token_probability: float | None
    minimum_generated_token_logprob: float | None
    p10_generated_token_logprob: float | None
    sequence_logprob_sum: float
    sequence_score_mean_logprob: float
    terminated_by_eos: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _linear_quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("quantile fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def whisper_compression_ratio(text: str) -> float:
    """Return Whisper's UTF-8 bytes / zlib-bytes repetition diagnostic."""

    payload = str(text).encode("utf-8")
    if not payload:
        return 0.0
    return len(payload) / len(zlib.compress(payload))


def summarize_whisper_confidence(
    token_ids: Sequence[int],
    token_logprobs: Sequence[float],
    *,
    decoded_text: str,
    eos_token_id: int,
    special_token_ids: Iterable[int],
) -> WhisperConfidence:
    """Summarize scored generation steps, trimming padding after the first EOS.

    ``token_ids`` must be the generated (not forced-prefix) token ids aligned
    one-to-one with ``token_logprobs`` returned by
    ``compute_transition_scores(..., normalize_logits=True)``.
    """

    if len(token_ids) != len(token_logprobs):
        raise ValueError("generated token ids and logprobs must align")
    if not token_ids:
        raise ValueError("Whisper generation has no scored steps")
    stop = len(token_ids)
    terminated = False
    for index, token_id in enumerate(token_ids):
        if int(token_id) == int(eos_token_id):
            stop = index + 1
            terminated = True
            break
    valid_ids = [int(value) for value in token_ids[:stop]]
    valid_scores = [float(value) for value in token_logprobs[:stop]]
    if any(not math.isfinite(value) for value in valid_scores):
        raise ValueError("Whisper transition scores must be finite")
    special = {int(value) for value in special_token_ids}
    content_scores = [
        score
        for token_id, score in zip(valid_ids, valid_scores, strict=True)
        if token_id not in special
    ]
    sequence_sum = float(sum(valid_scores))
    mean_logprob = (
        float(sum(content_scores) / len(content_scores))
        if content_scores
        else None
    )
    text_bytes = len(str(decoded_text).encode("utf-8"))
    eos_logprob = valid_scores[-1] if terminated else None
    return WhisperConfidence(
        compression_ratio=whisper_compression_ratio(decoded_text),
        content_token_count=len(content_scores),
        decoded_text_characters=len(str(decoded_text)),
        decoded_text_utf8_bytes=text_bytes,
        eos_logprob=eos_logprob,
        generated_step_count=len(valid_scores),
        max_length_reached=not terminated,
        mean_generated_token_logprob=mean_logprob,
        mean_generated_token_probability=(
            float(sum(math.exp(value) for value in content_scores) / len(content_scores))
            if content_scores
            else None
        ),
        minimum_generated_token_logprob=(min(content_scores) if content_scores else None),
        p10_generated_token_logprob=(
            _linear_quantile(content_scores, 0.10) if content_scores else None
        ),
        sequence_logprob_sum=sequence_sum,
        sequence_score_mean_logprob=sequence_sum / len(valid_scores),
        terminated_by_eos=terminated,
    )

