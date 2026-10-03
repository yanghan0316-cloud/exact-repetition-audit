#!/usr/bin/env python3
"""Verify the released n=4 aggregates without corpus text, audio, or models."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))
from analyze_ngram_postprimary_v1 import paired_bootstrap


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def equal(left, right, name):
    if isinstance(right, (dict, list)) and isinstance(left, str):
        left = json.loads(left)
    if isinstance(right, (float, int)) and not isinstance(right, bool):
        if not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{name}: {left} != {right}")
    elif left != right:
        raise AssertionError(f"{name}: {left!r} != {right!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = ROOT / "results/ngram_blocking"
    analysis = json.loads((results / "analysis.json").read_text(encoding="utf-8"))
    rows = read_csv(results / "per_meeting_counts.csv")
    fields = ("reference_chars", "substitutions", "deletions", "insertions", "errors")
    for row in rows:
        for field in fields:
            row[field] = int(row[field])
        row["cpcer"] = float(row["cpcer"])
        equal(row["errors"], sum(row[f] for f in ("substitutions", "deletions", "insertions")), "S+D+I")
        equal(row["cpcer"], row["errors"] / row["reference_chars"], "recording cpCER")
    equal(len(rows), 72, "recording-arm row count")
    for arm in "ABCD":
        selected = [r for r in rows if r["arm"] == arm]
        equal(sorted(r["session"] for r in selected), sorted(analysis["sessions"]), f"{arm} sessions")
        for field in fields:
            equal(sum(r[field] for r in selected), analysis["arms"][arm][field], f"{arm} {field}")
        equal(sum(r["errors"] for r in selected) / sum(r["reference_chars"] for r in selected), analysis["arms"][arm]["micro_cpcer"], f"{arm} micro cpCER")
        equal(sum(r["cpcer"] for r in selected) / len(selected), analysis["arms"][arm]["macro_cpcer"], f"{arm} macro cpCER")
    derived = paired_bootstrap(rows, repetitions=50000, seed=20261002)
    for comparison, values in derived.items():
        for key, value in values.items():
            equal(value, analysis["comparisons"][comparison][key], f"{comparison} {key}")
    for row in read_csv(results / "arm_summary.csv"):
        for key, value in row.items():
            if key == "arm":
                continue
            expected = analysis["arms"][row["arm"]][key]
            equal(value, expected, f"arm CSV {row['arm']} {key}")
    for row in read_csv(results / "paired_comparisons.csv"):
        comparison = row["comparison"]
        for key, value in row.items():
            if key == "comparison":
                continue
            equal(value, analysis["comparisons"][comparison][key], f"comparison CSV {comparison} {key}")
    for row in read_csv(ROOT / "figures/ngram_blocking/paired_effects.csv"):
        for key, value in row.items():
            if key != "comparison":
                equal(value, derived[row["comparison"]][key], f"paired figure {key}")
    for row in read_csv(ROOT / "figures/ngram_blocking/error_composition.csv"):
        left, right = row["comparison"].replace(" ", "").split("-")
        net = 0
        for field in ("substitutions", "deletions", "insertions"):
            delta = analysis["arms"][left][field] - analysis["arms"][right][field]
            equal(row[field + "_count_delta"], delta, f"composition {field} count")
            equal(row[field + "_percentage_points"], delta / analysis["reference_characters"] * 100, f"composition {field} pp")
            net += delta
        equal(row["net_percentage_points"], net / analysis["reference_characters"] * 100, "composition net")
    receipt = {
        "status": "PASS",
        "recording_arm_rows": len(rows),
        "arms": 4,
        "paired_contrasts": len(derived),
        "bootstrap_draws": 50000,
        "bootstrap_seed": 20261002,
        "checks": ["recording counts and cpCER", "arm totals and macro/micro cpCER", "recording-paired intervals and W/T/L", "arm and comparison CSVs", "figure CSV values"],
        "scope": "Aggregate numerical verification only. No acoustic rerun or listening revalidation. Text identity and residual event claims are supported by the recorded execution/analysis receipts, not inferred independently from count equality.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
