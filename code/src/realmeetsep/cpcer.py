"""Chinese cpCER and speaker-attributed CER utilities.

The default backend uses RapidFuzz's C++ unit-cost Levenshtein opcodes. A
deterministic pure-Python dynamic-programming implementation remains available
as the reference implementation and automatic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

try:
    import rapidfuzz as _rapidfuzz
    from rapidfuzz.distance import Levenshtein as _RapidFuzzLevenshtein
except ImportError:  # pragma: no cover - exercised in dependency-minimal installs
    _rapidfuzz = None
    _RapidFuzzLevenshtein = None

from .textgrid import Segment


NORMALIZATION_ID = "nfkc_casefold_unicode_letters_numbers_v1"
REFERENCE_POLICIES = ("linear_clip", "segment_overlap", "midpoint")
_ANNOTATION_RE = re.compile(r"<[^>]*>|\[[^\]]*\]|\{[^}]*\}")


def levenshtein_backend_info() -> dict[str, str]:
    """Describe the active standard-Levenshtein implementation."""

    if _RapidFuzzLevenshtein is not None:
        return {
            "name": "rapidfuzz",
            "version": str(_rapidfuzz.__version__),
            "implementation": "C++ Levenshtein.opcodes",
            "metric": "unit-cost Levenshtein",
        }
    return {
        "name": "python_reference",
        "version": "builtin_v1",
        "implementation": "dynamic programming",
        "metric": "unit-cost Levenshtein",
    }


def normalize_zh_text(text: str) -> str:
    """Normalize Mandarin transcripts to Unicode character scoring units."""

    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = _ANNOTATION_RE.sub("", value)
    return "".join(
        character
        for character in value
        if unicodedata.category(character)[0] in {"L", "N"}
    )


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_chars: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float | None:
        if self.reference_chars == 0:
            return None
        return self.errors / self.reference_chars

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_chars + other.reference_chars,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "reference_chars": self.reference_chars,
            "cer": self.cer,
            "levenshtein_backend": levenshtein_backend_info(),
        }


@dataclass(frozen=True)
class StreamTranscript:
    stream: str
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class PermutationScore:
    counts: ErrorCounts
    assignment: tuple[tuple[str, str | None], ...]


def _alignment_key(value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    substitutions, deletions, insertions, _ = value
    return substitutions + deletions + insertions, substitutions, deletions, insertions


def levenshtein_counts_python(reference: str, hypothesis: str) -> ErrorCounts:
    """Deterministic pure-Python DP reference implementation."""

    previous = [(0, 0, j, 0) for j in range(len(hypothesis) + 1)]
    for i, ref_character in enumerate(reference, start=1):
        current = [(0, i, 0, 0)]
        for j, hyp_character in enumerate(hypothesis, start=1):
            if ref_character == hyp_character:
                diagonal = previous[j - 1]
            else:
                s, d, ins, unused = previous[j - 1]
                diagonal = (s + 1, d, ins, unused)
            s, d, ins, unused = previous[j]
            deletion = (s, d + 1, ins, unused)
            s, d, ins, unused = current[j - 1]
            insertion = (s, d, ins + 1, unused)
            current.append(min((diagonal, deletion, insertion), key=_alignment_key))
        previous = current
    substitutions, deletions, insertions, _ = previous[-1]
    return ErrorCounts(substitutions, deletions, insertions, len(reference))


def levenshtein_counts_rapidfuzz(reference: str, hypothesis: str) -> ErrorCounts:
    """Count S/D/I from RapidFuzz C++ standard-Levenshtein opcodes."""

    if _RapidFuzzLevenshtein is None:
        raise RuntimeError("RapidFuzz is unavailable")
    substitutions = 0
    deletions = 0
    insertions = 0
    for opcode in _RapidFuzzLevenshtein.opcodes(reference, hypothesis):
        source_length = int(opcode.src_end - opcode.src_start)
        destination_length = int(opcode.dest_end - opcode.dest_start)
        if opcode.tag == "equal":
            continue
        if opcode.tag == "replace":
            common = min(source_length, destination_length)
            substitutions += common
            deletions += source_length - common
            insertions += destination_length - common
        elif opcode.tag == "delete":
            deletions += source_length
        elif opcode.tag == "insert":
            insertions += destination_length
        else:  # pragma: no cover - protects against an incompatible future API
            raise RuntimeError(f"unsupported RapidFuzz opcode: {opcode.tag!r}")
    return ErrorCounts(substitutions, deletions, insertions, len(reference))


def levenshtein_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Return S/D/I counts using C++ opcodes, with pure-Python fallback."""

    if _RapidFuzzLevenshtein is not None:
        return levenshtein_counts_rapidfuzz(reference, hypothesis)
    return levenshtein_counts_python(reference, hypothesis)


def _normalized_references(
    references: Mapping[str, str], *, already_normalized: bool
) -> list[tuple[str, str]]:
    return [
        (speaker, text if already_normalized else normalize_zh_text(text))
        for speaker, text in references.items()
    ]


def _normalized_hypotheses(
    hypotheses: Sequence[StreamTranscript], *, already_normalized: bool
) -> list[StreamTranscript]:
    return [
        StreamTranscript(
            stream=item.stream,
            speaker=item.speaker,
            text=item.text if already_normalized else normalize_zh_text(item.text),
        )
        for item in hypotheses
    ]


def score_cpcer(
    references: Mapping[str, str],
    hypotheses: Sequence[StreamTranscript],
    *,
    already_normalized: bool = False,
) -> PermutationScore:
    """Score generalized cpCER, padding unequal speaker counts with empties."""

    refs = _normalized_references(references, already_normalized=already_normalized)
    hyps = _normalized_hypotheses(hypotheses, already_normalized=already_normalized)
    size = max(len(refs), len(hyps))
    if size == 0:
        return PermutationScore(ErrorCounts(), ())
    if size > 8:
        raise ValueError("Exact cpCER permutation scoring is limited to 8 streams")

    refs = refs + [(f"__empty_reference_{i}", "") for i in range(size - len(refs))]
    hyps = hyps + [
        StreamTranscript(stream=f"__empty_hypothesis_{i}", text="", speaker=None)
        for i in range(size - len(hyps))
    ]
    matrix = [
        [levenshtein_counts(ref_text, hyp.text) for hyp in hyps]
        for _, ref_text in refs
    ]

    best_key: tuple | None = None
    best_counts: ErrorCounts | None = None
    best_assignment: tuple[tuple[str, str | None], ...] | None = None
    for order in permutations(range(size)):
        counts = ErrorCounts()
        assignment: list[tuple[str, str | None]] = []
        for ref_index, hyp_index in enumerate(order):
            counts += matrix[ref_index][hyp_index]
            ref_speaker = refs[ref_index][0]
            hyp_stream = hyps[hyp_index].stream
            if not ref_speaker.startswith("__empty_reference_"):
                assignment.append(
                    (
                        ref_speaker,
                        None if hyp_stream.startswith("__empty_hypothesis_") else hyp_stream,
                    )
                )
        key = (
            counts.errors,
            counts.substitutions,
            counts.deletions,
            counts.insertions,
            tuple(order),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_counts = counts
            best_assignment = tuple(assignment)
    assert best_counts is not None and best_assignment is not None
    return PermutationScore(best_counts, best_assignment)


def score_sa_cer(
    references: Mapping[str, str],
    hypotheses: Sequence[StreamTranscript],
    *,
    already_normalized: bool = False,
) -> ErrorCounts:
    """Score ordered speaker-attributed CER using explicit speaker labels."""

    refs = dict(_normalized_references(references, already_normalized=already_normalized))
    hyps = _normalized_hypotheses(hypotheses, already_normalized=already_normalized)
    grouped: dict[str, list[str]] = {}
    for item in hyps:
        if item.speaker is None:
            raise ValueError("SA-CER requires an explicit speaker on every hypothesis stream")
        grouped.setdefault(item.speaker, []).append(item.text)
    all_speakers = list(refs)
    all_speakers.extend(sorted(set(grouped) - set(refs)))
    total = ErrorCounts()
    for speaker in all_speakers:
        total += levenshtein_counts(refs.get(speaker, ""), "".join(grouped.get(speaker, [])))
    return total


def _linear_clip_segment(segment: Segment, start: float, end: float) -> str:
    text = normalize_zh_text(segment.text)
    if not text or segment.end <= start or segment.start >= end:
        return ""
    if segment.duration <= 0:
        return ""
    selected: list[str] = []
    for index, character in enumerate(text):
        centre = segment.start + (index + 0.5) * segment.duration / len(text)
        if start <= centre < end:
            selected.append(character)
    return "".join(selected)


def window_reference_text(
    segments: Iterable[Segment],
    *,
    start: float,
    end: float,
    policy: str = "linear_clip",
) -> str:
    """Construct a normalized speaker transcript for one analysis window."""

    if end <= start:
        raise ValueError("Window end must be greater than start")
    if policy not in REFERENCE_POLICIES:
        raise ValueError(f"Unknown reference policy {policy!r}; choose {REFERENCE_POLICIES}")
    output: list[str] = []
    for segment in sorted(segments):
        overlaps = segment.end > start and segment.start < end
        if not overlaps:
            continue
        if policy == "linear_clip":
            output.append(_linear_clip_segment(segment, start, end))
        elif policy == "segment_overlap":
            output.append(normalize_zh_text(segment.text))
        elif start <= (segment.start + segment.end) / 2.0 < end:
            output.append(normalize_zh_text(segment.text))
    return "".join(output)


def aggregate_counts(values: Iterable[ErrorCounts]) -> ErrorCounts:
    total = ErrorCounts()
    for value in values:
        total += value
    return total


__all__ = [
    "ErrorCounts",
    "NORMALIZATION_ID",
    "PermutationScore",
    "REFERENCE_POLICIES",
    "StreamTranscript",
    "aggregate_counts",
    "levenshtein_backend_info",
    "levenshtein_counts",
    "levenshtein_counts_python",
    "levenshtein_counts_rapidfuzz",
    "normalize_zh_text",
    "score_cpcer",
    "score_sa_cer",
    "window_reference_text",
]
