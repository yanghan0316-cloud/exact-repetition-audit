#!/usr/bin/env python3
"""Recompute released decoder and baseline aggregates from content-free counts.

This verifies accounting and cross-file consistency. It does not reconstruct
ASR, edit alignments, or reviewer judgments from the original audio.
"""
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with (ROOT / path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    if not math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12):
        raise AssertionError(f"numeric mismatch: {a} != {b}")


def counts(row, names):
    ref, error, deletion, substitution, insertion, rate = names
    total = sum(float(row[k]) for k in (deletion, substitution, insertion))
    close(total, row[error])
    close(total / float(row[ref]), row[rate])


def main():
    decoder = {}
    decoder_count = 0
    for corpus in ("aishell5", "misp2022"):
        data = rows(f"results/decoder_controls/{corpus}_recording_scores.csv")
        for row in data:
            counts(row, ("reference_chars", "cp_errors", "cp_deletions", "cp_substitutions", "cp_insertions", "meeting_concatenated_cpcer"))
            key = (corpus, row["session"], row["system"])
            if key in decoder:
                raise AssertionError("duplicate decoder row")
            decoder[key] = row
        decoder_count += len(data)
    aggregate_count = 0
    for row in rows("results/decoder_controls/RESULTS.csv"):
        selected = [value for (corpus, session, system), value in decoder.items()
                    if corpus == row["corpus"] and system == row["system"]
                    and (row["stratum"] == "pooled" or session.startswith("A5E" + row["stratum"][-1] + "_"))]
        if len(selected) != int(row["sessions"]):
            raise AssertionError("decoder meeting membership mismatch")
        close(sum(int(x["cp_errors"]) for x in selected) / sum(int(x["reference_chars"]) for x in selected), row["micro_cpcer"])
        aggregate_count += 1
    baseline = defaultdict(list)
    for row in rows("results/direct_baselines/meeting_metrics.csv"):
        counts(row, ("reference_characters", "cp_errors", "cp_deletions", "cp_substitutions", "cp_insertions", "cpcer"))
        close(int(row["input_unicode_characters"]) - int(row["output_unicode_characters"]), row["deleted_unicode_characters"])
        baseline[tuple(row[k] for k in ("corpus", "reference_cell", "decoder_condition", "source", "method"))].append(row)
    for row in rows("results/direct_baselines/aggregate_metrics.csv"):
        key = tuple(row[k] for k in ("corpus", "reference_cell", "decoder_condition", "source", "method"))
        selected = baseline[key]
        if len(selected) != int(row["meetings"]):
            raise AssertionError("baseline membership mismatch")
        for field in ("cp_errors", "cp_substitutions", "cp_deletions", "cp_insertions", "reference_characters", "input_unicode_characters", "output_unicode_characters", "deleted_unicode_characters", "exact_event_count", "windows"):
            close(sum(int(x[field]) for x in selected), row[field])
        close(sum(int(x["cp_errors"]) for x in selected) / sum(int(x["reference_characters"]) for x in selected), row["micro_cpcer"])
    matched = rows("results/matched_deletion/meeting_scores_content_free.csv")
    for row in matched:
        counts(row, ("reference_characters", "errors", "deletions", "substitutions", "insertions", "cpcer"))
        close(float(row["input_raw_characters"]) - float(row["output_raw_characters"]), row["removed_raw_characters"])
    traces = 0
    with gzip.open(ROOT / "results/direct_baselines/content_free_event_audit.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            close(row["input_unicode_characters"] - row["output_unicode_characters"], row["deleted_unicode_characters"])
            if row["exact_event_count"] != len(row["events"]):
                raise AssertionError("trace event count mismatch")
            if not row["modified"] and row["input_text_sha256"] != row["output_text_sha256"]:
                raise AssertionError("no-event identity mismatch")
            if row["method"] in ("existing_canonical_guard_keep2", "ltr_greedy_character_keep2") and row["residual_v035_exact_event_count"]:
                raise AssertionError("character method left an eligible event")
            traces += 1
    if traces != 80256:
        raise AssertionError("direct baseline trace inventory differs")
    print(json.dumps({"status": "PASS", "decoder_meeting_rows": decoder_count, "decoder_aggregate_rows": aggregate_count, "baseline_aggregate_cells": len(baseline), "matched_deletion_rows": len(matched), "method_traces": traces}, sort_keys=True))


if __name__ == "__main__":
    main()
