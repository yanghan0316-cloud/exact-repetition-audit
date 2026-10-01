"""Efficient exact recording-bootstrap representation for v046 diagnostics.

Selecting ``n`` recordings with replacement is distributionally identical to
drawing their selection-count vector from ``Multinomial(n; 1/n, ..., 1/n)``.
This implementation uses that count vector and matrix multiplication while
preserving the frozen unit, draws, estimands, and derived seeds.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .postprimary_alignment_retention_v046 import (
    AlignmentCounts,
    aggregate_alignment_counts,
)


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


def paired_micro_bootstrap_multinomial_v046(
    comparator: Sequence[AlignmentCounts],
    candidate: Sequence[AlignmentCounts],
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    """Paired candidate-minus-comparator micro-rate bootstrap."""

    if len(comparator) != len(candidate) or not comparator:
        raise ValueError("paired inputs must be non-empty and have equal lengths")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
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
    recording_count = len(comparator)
    weights = rng.multinomial(
        recording_count,
        np.full(recording_count, 1.0 / recording_count),
        size=draws,
    )
    base_ref_sum = weights @ base_ref
    new_ref_sum = weights @ new_ref
    base_hyp_sum = weights @ base_hyp
    new_hyp_sum = weights @ new_hyp
    with np.errstate(divide="ignore", invalid="ignore"):
        cpcer_samples = (
            (weights @ new_err) / new_ref_sum
            - (weights @ base_err) / base_ref_sum
        )
        amrr_samples = (
            (weights @ new_c) / new_ref_sum
            - (weights @ base_c) / base_ref_sum
        )
        amhr_samples = (
            (weights @ new_c) / new_hyp_sum
            - (weights @ base_c) / base_hyp_sum
        )

    return {
        "direction": "candidate_minus_comparator",
        "recordings": recording_count,
        "draws": draws,
        "seed": seed,
        "bootstrap_representation": (
            "multinomial_recording_selection_counts_exact_nonparametric_bootstrap"
        ),
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


__all__ = ["paired_micro_bootstrap_multinomial_v046"]
