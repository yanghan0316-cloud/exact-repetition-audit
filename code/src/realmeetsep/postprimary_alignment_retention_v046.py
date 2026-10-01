"""Post-primary alignment-derived retention diagnostics (v046).

These helpers intentionally do not define semantic recall or precision.
``matched_characters`` is the number of equality operations implied by one
particular unit-cost Levenshtein alignment after one particular stream
assignment.  It is therefore backend-, tie-break-, and assignment-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import numpy as np

from .cpcer import ErrorCounts, aggregate_counts, levenshtein_counts


SCHEMA = "postprimary_alignment_retention_metrics_v046"


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class AlignmentCounts:
    """S/D/I counts plus alignment-derived matched and length quantities."""

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_chars: int = 0

    def __post_init__(self) -> None:
        for name in (
            "substitutions",
            "deletions",
            "insertions",
            "reference_chars",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        if self.substitutions + self.deletions > self.reference_chars:
            raise ValueError(
                "invalid alignment counts: substitutions + deletions exceeds "
                "reference_chars"
            )
        if self.matched_chars + self.substitutions + self.insertions < 0:
            raise ValueError("invalid alignment counts yield negative hypothesis length")

    @classmethod
    def from_error_counts(cls, counts: ErrorCounts) -> "AlignmentCounts":
        return cls(
            substitutions=int(counts.substitutions),
            deletions=int(counts.deletions),
            insertions=int(counts.insertions),
            reference_chars=int(counts.reference_chars),
        )

    @classmethod
    def from_cpcer_row(cls, row: Mapping[str, object]) -> "AlignmentCounts":
        return cls(
            substitutions=int(str(row["cp_substitutions"])),
            deletions=int(str(row["cp_deletions"])),
            insertions=int(str(row["cp_insertions"])),
            reference_chars=int(str(row["reference_chars"])),
        )

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def matched_chars(self) -> int:
        return self.reference_chars - self.substitutions - self.deletions

    @property
    def hypothesis_chars(self) -> int:
        # Both identities are checked by ``assert_invariants``.
        return self.matched_chars + self.substitutions + self.insertions

    @property
    def cpcer(self) -> float | None:
        return _safe_ratio(self.errors, self.reference_chars)

    @property
    def alignment_matched_reference_rate(self) -> float | None:
        return _safe_ratio(self.matched_chars, self.reference_chars)

    @property
    def alignment_matched_hypothesis_rate(self) -> float | None:
        return _safe_ratio(self.matched_chars, self.hypothesis_chars)

    def assert_invariants(self) -> None:
        if self.reference_chars != (
            self.matched_chars + self.substitutions + self.deletions
        ):
            raise AssertionError("N_ref != C + S + D")
        if self.hypothesis_chars != (
            self.matched_chars + self.substitutions + self.insertions
        ):
            raise AssertionError("N_hyp != C + S + I")
        if self.hypothesis_chars != self.reference_chars - self.deletions + self.insertions:
            raise AssertionError("N_hyp != N_ref - D + I")
        if self.errors != self.substitutions + self.deletions + self.insertions:
            raise AssertionError("errors != S + D + I")

    def __add__(self, other: "AlignmentCounts") -> "AlignmentCounts":
        if not isinstance(other, AlignmentCounts):
            return NotImplemented
        return AlignmentCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_chars=self.reference_chars + other.reference_chars,
        )

    def to_dict(self) -> dict[str, int | float | None]:
        self.assert_invariants()
        return {
            "matched_characters": self.matched_chars,
            "reference_characters": self.reference_chars,
            "hypothesis_characters": self.hypothesis_chars,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "micro_cpcer": self.cpcer,
            "alignment_matched_reference_rate": (
                self.alignment_matched_reference_rate
            ),
            "alignment_matched_hypothesis_rate": (
                self.alignment_matched_hypothesis_rate
            ),
        }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def aggregate_alignment_counts(
    values: Sequence[AlignmentCounts],
) -> AlignmentCounts:
    total = AlignmentCounts()
    for value in values:
        total += value
    total.assert_invariants()
    return total


def score_fixed_stream_assignment(
    references: Mapping[str, str],
    hypotheses: Mapping[str, str],
    assignment: Mapping[str, str | None],
) -> AlignmentCounts:
    """Score hypotheses under a supplied reference-to-stream assignment.

    Inputs must already use the scorer's normalized character units.  Every
    real reference speaker must occur exactly once in ``assignment``.  Any
    hypothesis stream not assigned to a real reference is scored against an
    empty reference, matching generalized cpCER padding semantics.
    """

    if set(assignment) != set(references):
        missing = sorted(set(references) - set(assignment))
        extra = sorted(set(assignment) - set(references))
        raise ValueError(
            f"assignment/reference speaker mismatch: missing={missing}, extra={extra}"
        )
    used_streams: set[str] = set()
    parts: list[ErrorCounts] = []
    for speaker, reference_text in references.items():
        stream = assignment[speaker]
        if stream is None:
            hypothesis_text = ""
        else:
            if stream not in hypotheses:
                raise ValueError(f"assigned stream absent from hypotheses: {stream!r}")
            if stream in used_streams:
                raise ValueError(f"stream assigned more than once: {stream!r}")
            used_streams.add(stream)
            hypothesis_text = hypotheses[stream]
        parts.append(levenshtein_counts(reference_text, hypothesis_text))
    for stream in sorted(set(hypotheses) - used_streams):
        parts.append(levenshtein_counts("", hypotheses[stream]))
    result = AlignmentCounts.from_error_counts(aggregate_counts(parts))
    result.assert_invariants()
    return result


def stable_bootstrap_seed(base_seed: int, *parts: str) -> int:
    _require_nonnegative_int("base_seed", base_seed)
    namespace = "|".join(parts)
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def _metric_summary(
    observed: float | None,
    samples: np.ndarray | None,
    *,
    draws: int,
) -> dict[str, object]:
    if observed is None or samples is None:
        return {
            "observed": None,
            "percentile_95_ci": None,
            "valid_draws": 0,
            "undefined_draws": draws,
        }
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return {
            "observed": observed,
            "percentile_95_ci": None,
            "valid_draws": 0,
            "undefined_draws": draws,
        }
    return {
        "observed": float(observed),
        "percentile_95_ci": [
            float(np.quantile(finite, 0.025)),
            float(np.quantile(finite, 0.975)),
        ],
        "valid_draws": int(finite.size),
        "undefined_draws": int(draws - finite.size),
    }


def paired_micro_bootstrap(
    comparator: Sequence[AlignmentCounts],
    candidate: Sequence[AlignmentCounts],
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Paired recording bootstrap of candidate-minus-comparator micro rates."""

    if len(comparator) != len(candidate) or not comparator:
        raise ValueError("paired inputs must be non-empty and have equal lengths")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    _require_nonnegative_int("seed", seed)
    for base, new in zip(comparator, candidate):
        base.assert_invariants()
        new.assert_invariants()
        if base.reference_chars != new.reference_chars:
            raise ValueError("paired recording reference denominators differ")

    def array(name: str, values: Sequence[AlignmentCounts]) -> np.ndarray:
        return np.asarray([getattr(value, name) for value in values], dtype=np.int64)

    base_c = array("matched_chars", comparator)
    new_c = array("matched_chars", candidate)
    base_ref = array("reference_chars", comparator)
    new_ref = array("reference_chars", candidate)
    base_hyp = array("hypothesis_chars", comparator)
    new_hyp = array("hypothesis_chars", candidate)
    base_err = array("errors", comparator)
    new_err = array("errors", candidate)

    base_total = aggregate_alignment_counts(comparator)
    new_total = aggregate_alignment_counts(candidate)
    observed_cpcer = (
        None
        if base_total.cpcer is None or new_total.cpcer is None
        else new_total.cpcer - base_total.cpcer
    )
    observed_amrr = (
        None
        if base_total.alignment_matched_reference_rate is None
        or new_total.alignment_matched_reference_rate is None
        else new_total.alignment_matched_reference_rate
        - base_total.alignment_matched_reference_rate
    )
    observed_amhr = (
        None
        if base_total.alignment_matched_hypothesis_rate is None
        or new_total.alignment_matched_hypothesis_rate is None
        else new_total.alignment_matched_hypothesis_rate
        - base_total.alignment_matched_hypothesis_rate
    )

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(comparator), size=(draws, len(comparator)))
    base_ref_sum = base_ref[indices].sum(axis=1)
    new_ref_sum = new_ref[indices].sum(axis=1)
    base_hyp_sum = base_hyp[indices].sum(axis=1)
    new_hyp_sum = new_hyp[indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cpcer_samples = (
            new_err[indices].sum(axis=1) / new_ref_sum
            - base_err[indices].sum(axis=1) / base_ref_sum
        )
        amrr_samples = (
            new_c[indices].sum(axis=1) / new_ref_sum
            - base_c[indices].sum(axis=1) / base_ref_sum
        )
        amhr_samples = (
            new_c[indices].sum(axis=1) / new_hyp_sum
            - base_c[indices].sum(axis=1) / base_hyp_sum
        )

    return {
        "direction": "candidate_minus_comparator",
        "recordings": len(comparator),
        "draws": draws,
        "seed": seed,
        "delta_matched_characters": new_total.matched_chars - base_total.matched_chars,
        "delta_hypothesis_characters": (
            new_total.hypothesis_chars - base_total.hypothesis_chars
        ),
        "micro_cpcer": _metric_summary(
            observed_cpcer, cpcer_samples, draws=draws
        ),
        "alignment_matched_reference_rate": _metric_summary(
            observed_amrr, amrr_samples, draws=draws
        ),
        "alignment_matched_hypothesis_rate": _metric_summary(
            observed_amhr, amhr_samples, draws=draws
        ),
    }


__all__ = [
    "AlignmentCounts",
    "SCHEMA",
    "aggregate_alignment_counts",
    "paired_micro_bootstrap",
    "score_fixed_stream_assignment",
    "stable_bootstrap_seed",
]
