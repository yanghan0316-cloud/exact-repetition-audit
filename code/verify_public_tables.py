#!/usr/bin/env python3
"""Recompute public Eval36 scores, paired intervals, and the empty anchor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SYSTEMS = (
    "raw_cfg_low_packet_gated",
    "guarded_cfg_low_packet_gated",
    "raw_public_sortformer_v2_low_packet_gated",
    "guarded_public_sortformer_v2_low_packet_gated",
)
CONTRASTS = {
    "v035_guard_effect": (
        "guarded_cfg_low_packet_gated",
        "raw_cfg_low_packet_gated",
    ),
    "public_guard_effect": (
        "guarded_public_sortformer_v2_low_packet_gated",
        "raw_public_sortformer_v2_low_packet_gated",
    ),
    "guarded_learned_vs_guarded_public": (
        "guarded_cfg_low_packet_gated",
        "guarded_public_sortformer_v2_low_packet_gated",
    ),
    "raw_learned_vs_raw_public": (
        "raw_cfg_low_packet_gated",
        "raw_public_sortformer_v2_low_packet_gated",
    ),
}
TRIGGERED = "TRIGGERED_EVENT"
CONTROL = "UNTRIGGERED_CONTROL"
CORRECT = "GUARD_CORRECT_ARTIFACT_REMOVAL"
FALSE_POSITIVE = "GUARD_FALSE_POSITIVE_SPEECH_REMOVAL"
CONTROL_CLEAR = "CONTROL_NO_OBVIOUS_REPETITION"
CONTROL_MISSED = "CONTROL_MISSED_REPETITION_ARTIFACT"
LABEL_FIELDS = (
    "review_id",
    "selection_kind",
    "split",
    "arm",
    "reviewer_a_label",
    "reviewer_a_confidence",
    "reviewer_b_label",
    "reviewer_b_confidence",
    "exact_agreement",
)


def close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{label}: {observed} != {expected}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


def verify_scores(root: Path) -> tuple[dict[tuple[str, str], dict[str, float | int]], list[str]]:
    summary = json.loads(
        (root / "results/eval36_summary.json").read_text(encoding="utf-8")
    )
    _, rows = read_csv(root / "results/eval36_recording_system_cpcer.csv")
    if len(rows) != 144:
        raise AssertionError("expected 144 score rows")
    table: dict[tuple[str, str], dict[str, float | int]] = {}
    for row in rows:
        session, system = row["session"], row["system"]
        if system not in SYSTEMS or (session, system) in table:
            raise AssertionError("unexpected or duplicate score row")
        value: dict[str, float | int] = {
            key: int(row[key])
            for key in (
                "reference_chars",
                "deletions",
                "substitutions",
                "insertions",
                "errors",
            )
        }
        value["cpcer"] = float(row["cpcer"])
        if value["errors"] != (
            value["deletions"] + value["substitutions"] + value["insertions"]
        ):
            raise AssertionError("D/S/I mismatch")
        close(
            float(value["cpcer"]),
            int(value["errors"]) / int(value["reference_chars"]),
            "row cpCER",
        )
        table[(session, system)] = value
    sessions = sorted({session for session, _ in table})
    if len(sessions) != 36 or set(table) != {
        (session, system) for session in sessions for system in SYSTEMS
    }:
        raise AssertionError("expected 36 meetings x 4 systems")
    for session in sessions:
        refs = {int(table[(session, system)]["reference_chars"]) for system in SYSTEMS}
        if len(refs) != 1:
            raise AssertionError("reference count differs across systems")
    for system in SYSTEMS:
        values = [table[(session, system)] for session in sessions]
        expected = summary["pooled_four_system_scores"][system]
        refs = sum(int(value["reference_chars"]) for value in values)
        errors = sum(int(value["errors"]) for value in values)
        if refs != expected["reference_characters"] or errors != expected["errors"]:
            raise AssertionError("pooled count mismatch")
        for field in ("deletions", "substitutions", "insertions"):
            if sum(int(value[field]) for value in values) != expected[field]:
                raise AssertionError(f"pooled {field} mismatch")
        close(errors / refs, expected["micro_cpcer"], "pooled micro")
        close(
            sum(float(value["cpcer"]) for value in values) / len(values),
            expected["macro_cpcer"],
            "pooled macro",
        )
    strata = {
        "eval1_primary": [s for s in sessions if s.startswith("A5E1_")],
        "eval2_exact_replication": [s for s in sessions if s.startswith("A5E2_")],
        "pooled_supportive": sessions,
    }
    for stratum, selected in strata.items():
        for contrast, (candidate, comparator) in CONTRASTS.items():
            expected = summary["paired_analyses"][stratum][contrast]
            refs = np.asarray(
                [table[(s, candidate)]["reference_chars"] for s in selected],
                dtype=np.int64,
            )
            delta = np.asarray(
                [
                    table[(s, candidate)]["errors"]
                    - table[(s, comparator)]["errors"]
                    for s in selected
                ],
                dtype=np.int64,
            )
            per = delta / refs
            rng = np.random.default_rng(int(expected["bootstrap_seed"]))
            indices = rng.integers(
                0,
                len(selected),
                size=(int(expected["bootstrap_replicates"]), len(selected)),
            )
            micro = delta[indices].sum(1) / refs[indices].sum(1)
            macro = per[indices].mean(1)
            close(delta.sum() / refs.sum(), expected["micro_cpcer_delta"], "paired micro")
            close(float(per.mean()), expected["macro_mean_cpcer_delta"], "paired macro")
            for observed, wanted in zip(
                np.quantile(micro, [0.025, 0.975]), expected["micro_delta_95ci"]
            ):
                close(float(observed), float(wanted), "micro CI")
            for observed, wanted in zip(
                np.quantile(macro, [0.025, 0.975]), expected["macro_delta_95ci"]
            ):
                close(float(observed), float(wanted), "macro CI")
            outcomes = (
                int((delta < 0).sum()),
                int((delta == 0).sum()),
                int((delta > 0).sum()),
            )
            if outcomes != (expected["wins"], expected["ties"], expected["losses"]):
                raise AssertionError("meeting outcomes mismatch")
            close(
                float(np.mean(micro >= 0)),
                expected["micro_probability_ge_zero"],
                "bootstrap tail",
            )
    return table, sessions


def verify_empty_anchor(
    root: Path,
    table: dict[tuple[str, str], dict[str, float | int]],
    sessions: list[str],
) -> None:
    fields, rows = read_csv(root / "results/empty_output_baseline.csv")
    if fields != (
        "stratum",
        "reference_characters",
        "substitutions",
        "deletions",
        "insertions",
        "errors",
        "cpcer",
    ) or len(rows) != 3:
        raise AssertionError("unexpected empty-output table")
    expected_sessions = {
        "Eval1": [session for session in sessions if session.startswith("A5E1_")],
        "Eval2": [session for session in sessions if session.startswith("A5E2_")],
        "Pooled": sessions,
    }
    if [row["stratum"] for row in rows] != ["Eval1", "Eval2", "Pooled"]:
        raise AssertionError("unexpected empty-output strata")
    for row in rows:
        stratum = row["stratum"]
        reference = sum(
            int(table[(session, SYSTEMS[0])]["reference_chars"])
            for session in expected_sessions[stratum]
        )
        if (
            int(row["reference_characters"]) != reference
            or int(row["deletions"]) != reference
            or int(row["errors"]) != reference
            or int(row["substitutions"]) != 0
            or int(row["insertions"]) != 0
        ):
            raise AssertionError("empty-output count mismatch")
        close(float(row["cpcer"]), 1.0, "empty-output cpCER")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    table, sessions = verify_scores(root)
    verify_empty_anchor(root, table, sessions)
    print(json.dumps({"status": "PASS", "meetings": 36, "score_rows": 144, "contrasts": 12, "empty_anchor_rows": 3}, sort_keys=True))

if __name__ == "__main__":
    main()
