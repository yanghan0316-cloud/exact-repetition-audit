#!/usr/bin/env python
"""Assemble and score the prespecified Primary/public n=4 comparison.

The existing V4/V5 files are read only.  A/B-only output is explicitly partial;
provide --c-dir to require all C windows and produce the four-arm comparison.
All scoring and text normalization reuse score_meeting_cpcer.py unchanged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
import score_meeting_cpcer as legacy_score  # noqa: E402
from realmeetsep.asr_repetition_guard_v035 import guard_repetitions  # noqa: E402
from realmeetsep.cpcer import levenshtein_backend_info  # noqa: E402

EXPERIMENT = PROJECT / "experiments/whisper_ngram_blocking_postprimary_v1"
V4 = PROJECT / "experiments/whisper_control_comparison_postprimary_v4/aishell5"
V5 = PROJECT / "experiments/whisper_control_comparison_postprimary_v5/aishell5"
SOURCE_SYSTEM = "raw_public_sortformer_v2_low_packet_gated"
V5_SYSTEMS = {
    "A": "control_cap224_fallback_public_sortformer_v2_low_packet_gated",
    "B": "control_cap224_fallback_guarded_public_sortformer_v2_low_packet_gated",
}
GUARD_PARAMS = {
    "max_period_chars": 12,
    "min_repetitions": 6,
    "min_span_chars": 24,
    "keep_repetitions": 2,
}
COMPARISONS = (("C", "A"), ("D", "C"), ("C", "B"), ("D", "B"))
METADATA = ("schema", "asr_model", "split", "session", "slot", "meeting_stream",
            "segment_index", "start", "end")


def read_jsonl(path: Path) -> list[dict]:
    return legacy_score.read_jsonl(path)


def key(row: dict) -> tuple[str, int, int]:
    return str(row["session"]), int(row["slot"]), int(row["segment_index"])


def index_rows(rows: list[dict], label: str) -> dict[tuple, dict]:
    output = {}
    for row in rows:
        item_key = key(row)
        if item_key in output:
            raise ValueError(f"{label}: duplicate window {item_key}")
        output[item_key] = row
    return output


def select_rows(rows: list[dict], sessions: set[str], system: str) -> list[dict]:
    return [r for r in rows if r["session"] in sessions and r["system"] == system]


def assert_same_windows(expected: list[dict], actual: list[dict], label: str,
                        *, compare_text: bool = False) -> None:
    left, right = index_rows(expected, "expected"), index_rows(actual, label)
    if set(left) != set(right):
        missing, extra = sorted(set(left) - set(right)), sorted(set(right) - set(left))
        raise ValueError(f"{label}: window coverage mismatch: missing={missing[:5]} "
                         f"({len(missing)}), extra={extra[:5]} ({len(extra)})")
    fields = METADATA + (("text",) if compare_text else ())
    for item_key, row in left.items():
        for field in fields:
            if row.get(field) != right[item_key].get(field):
                raise ValueError(f"{label}: {field} differs at {item_key}")


def validate_audio_coverage(audio: list[dict], rows: list[dict]) -> None:
    left, right = index_rows(audio, "audio manifest"), index_rows(rows, "A")
    if set(left) != set(right):
        raise ValueError("A differs from the complete Primary/public audio inventory")
    for item_key, row in left.items():
        for field in ("session", "slot", "segment_index", "meeting_stream", "start", "end"):
            if row[field] != right[item_key][field]:
                raise ValueError(f"A/audio {field} differs at {item_key}")


def guarded_rows(rows: list[dict], arm: str) -> tuple[list[dict], list[dict]]:
    transformed, traces = [], []
    for row in rows:
        result = guard_repetitions(row["text"], **GUARD_PARAMS)
        transformed.append(dict(row, system=arm, text=result.text))
        if result.events:
            traces.append({
                "arm": arm, "session": row["session"], "slot": row["slot"],
                "segment_index": row["segment_index"], "input_text": row["text"],
                "output_text": result.text, **result.audit_dict(),
            })
    return transformed, traces


def morphology(rows: list[dict], source_rows: list[dict],
               baseline_raw: int, baseline_normalized: int) -> tuple[dict, list[dict]]:
    raw_chars = normalized_chars = raw_empty = normalized_empty = 0
    residual_events = residual_windows = 0
    residual_traces = []
    for row in rows:
        text = row["text"]
        normalized = legacy_score.normalize_zh_text(text)
        raw_chars += len(text)
        normalized_chars += len(normalized)
        raw_empty += int(text == "")
        normalized_empty += int(normalized == "")
        result = guard_repetitions(text, **GUARD_PARAMS)
        residual_events += len(result.events)
        residual_windows += int(bool(result.events))
        if result.events:
            residual_traces.append({
                "arm": row["system"], "session": row["session"],
                "slot": row["slot"], "segment_index": row["segment_index"],
                "final_text": text, "canonical_fixed_point_text": result.text,
                "ordered_edit_trace": [asdict(event) for event in result.events],
            })
    source_raw = sum(len(row["text"]) for row in source_rows)
    source_normalized = sum(len(legacy_score.normalize_zh_text(row["text"]))
                            for row in source_rows)
    return {
        "window_count": len(rows), "raw_characters": raw_chars,
        "normalized_characters": normalized_chars,
        "raw_length_ratio_to_A": raw_chars / baseline_raw if baseline_raw else None,
        "normalized_length_ratio_to_A": (normalized_chars / baseline_normalized
                                          if baseline_normalized else None),
        "raw_empty_windows": raw_empty, "normalized_empty_windows": normalized_empty,
        "residual_canonical_guard_events": residual_events,
        "residual_canonical_guard_windows": residual_windows,
        "guard_removed_raw_characters": source_raw - raw_chars,
        "guard_removed_normalized_characters": source_normalized - normalized_chars,
    }, residual_traces


def score_arms(arms: dict[str, list[dict]], reference_rows: list[dict]) -> tuple[dict, list[dict]]:
    references, split = legacy_score.validate_reference_rows(reference_rows)
    for reference in references.values():
        for text in reference["references"].values():
            if legacy_score.normalize_zh_text(text) != text:
                raise ValueError("Legacy full-meeting reference must already be normalized")
    flat_rows = [r for rows in arms.values() for r in rows]
    streams, asr_model = legacy_score.build_meeting_streams(flat_rows, references, split)
    by_arm, details = {}, []
    for arm in sorted(streams):
        counts = []
        for session in sorted(references):
            cp = legacy_score.score_cpcer(references[session]["references"],
                                          streams[arm][session], already_normalized=True)
            counts.append(cp.counts)
            details.append({
                "arm": arm, "session": session, "asr_model": asr_model,
                "reference_chars": cp.counts.reference_chars,
                "substitutions": cp.counts.substitutions,
                "deletions": cp.counts.deletions, "insertions": cp.counts.insertions,
                "errors": cp.counts.errors, "cpcer": cp.counts.cer,
                "cp_assignment": dict(cp.assignment),
            })
        total = legacy_score.aggregate_counts(counts)
        by_arm[arm] = {
            "status": "available", "meetings": len(counts), **total.to_dict(),
            "micro_cpcer": total.cer,
            "substitution_rate": total.substitutions / total.reference_chars,
            "deletion_rate": total.deletions / total.reference_chars,
            "insertion_rate": total.insertions / total.reference_chars,
            "macro_cpcer": float(np.mean([v.cer for v in counts])),
        }
    return by_arm, details


def paired_bootstrap(details: list[dict], *, repetitions: int = 50000,
                     seed: int = 20261002) -> dict[str, dict]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    groups = defaultdict(dict)
    for row in details:
        if row["session"] in groups[row["arm"]]:
            raise ValueError("duplicate meeting in bootstrap input")
        groups[row["arm"]][row["session"]] = row
    sessions = sorted(next(iter(groups.values())))
    if any(set(group) != set(sessions) for group in groups.values()):
        raise ValueError("bootstrap arms must contain the same meetings")
    samples = np.random.default_rng(seed).integers(0, len(sessions),
                                                  size=(repetitions, len(sessions)))
    result = {}
    for left, right in COMPARISONS:
        label = f"{left}-{right}"
        if left not in groups or right not in groups:
            result[label] = {"status": "unavailable", "reason": "C/D have not been decoded"}
            continue
        a = [groups[left][s] for s in sessions]
        b = [groups[right][s] for s in sessions]
        if any(x["reference_chars"] != y["reference_chars"] for x, y in zip(a, b)):
            raise ValueError("paired bootstrap reference denominators differ")
        denominator = np.asarray([r["reference_chars"] for r in a], dtype=np.int64)
        if np.any(denominator <= 0):
            raise ValueError("paired bootstrap requires positive reference denominators")
        errors_left = np.asarray([r["errors"] for r in a], dtype=np.int64)
        errors_right = np.asarray([r["errors"] for r in b], dtype=np.int64)
        difference = errors_left - errors_right
        # Recompute each resampled micro numerator and denominator. Never average CERs.
        sampled_delta = (difference[samples].sum(axis=1) /
                         denominator[samples].sum(axis=1))
        low, high = np.percentile(sampled_delta, [2.5, 97.5], method="linear")
        point = int(difference.sum()) / int(denominator.sum())
        result[label] = {
            "status": "available", "delta_micro_cpcer": point,
            "delta_percentage_points": point * 100,
            "ci95_lower": float(low), "ci95_upper": float(high),
            "ci95_lower_percentage_points": float(low * 100),
            "ci95_upper_percentage_points": float(high * 100),
            "wins": int(np.sum(difference < 0)), "ties": int(np.sum(difference == 0)),
            "losses": int(np.sum(difference > 0)), "meetings": len(sessions),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
            "unit": "meeting", "method": "paired percentile; resampled sum(errors)/sum(ref)",
        }
    return result


def verify_v5_scores(details: list[dict], path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        historical = {(r["system"], r["session"]): r for r in csv.DictReader(handle)}
    checked = 0
    for row in details:
        if row["arm"] not in V5_SYSTEMS:
            continue
        old = historical[(V5_SYSTEMS[row["arm"]], row["session"])]
        for current, prior in (("substitutions", "cp_substitutions"),
                               ("deletions", "cp_deletions"),
                               ("insertions", "cp_insertions"),
                               ("errors", "cp_errors"), ("reference_chars", "reference_chars")):
            if row[current] != int(old[prior]):
                raise ValueError(f"V5 score mismatch {row['arm']}/{row['session']}: {current}")
        if row["cp_assignment"] != json.loads(old["cp_assignment"]):
            raise ValueError(f"V5 assignment mismatch {row['arm']}/{row['session']}")
        checked += 1
    return {"status": "passed", "checked_arm_meetings": checked,
            "counts_and_assignments_exact_match": True, "source": str(path.resolve())}


def validate_c_run(directory: Path) -> dict:
    path = directory / "decoder_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    required = {"ngram_size": 4, "max_new_tokens": 224, "partial_smoke": False,
                "decoded_records": 1744, "decoded_batches": 222, "decoded_sessions": 18}
    for field, expected in required.items():
        if summary.get(field) != expected:
            raise ValueError(f"C decoder summary {field} must be {expected!r}; "
                             f"got {summary.get(field)!r}")
    return {"status": "passed", "source": str(path.resolve()), **required}


def decode_diagnostics(directory: Path, sessions: set[str], expected: list[dict],
                       *, baseline: bool) -> tuple[dict, list[dict]]:
    t0_path = directory / ("cap224_t0_scores.jsonl" if baseline else "t0_scores.jsonl")
    attempt_path = directory / "temperature_fallback_attempts.jsonl"
    if not t0_path.exists() or not attempt_path.exists():
        raise ValueError(f"Missing required decode diagnostics in {directory}")
    t0_rows = select_rows(read_jsonl(t0_path), sessions, SOURCE_SYSTEM)
    t0_index = index_rows(t0_rows, "T0 diagnostics")
    if set(t0_index) != set(index_rows(expected, "hypotheses")):
        raise ValueError("T0 diagnostics must cover every hypothesis window")
    attempts = select_rows(read_jsonl(attempt_path), sessions, SOURCE_SYSTEM)
    by_key = defaultdict(list)
    for attempt in attempts:
        if key(attempt) not in t0_index:
            raise ValueError("Fallback attempt absent from T0 inventory")
        by_key[key(attempt)].append(attempt)
    final_rows = []
    for item_key, row in t0_index.items():
        triggered = bool(row.get("compression_ratio_gt_2_4",
                                 row.get("triggered_fallback", row.get("fallback_triggered", False))))
        chain = sorted(by_key[item_key], key=lambda a: float(a["temperature"]))
        if triggered != bool(chain):
            raise ValueError(f"Fallback trigger/attempt mismatch: {item_key}")
        if chain:
            expected_temperatures = [0.2, 0.4, 0.6, 0.8, 1.0][:len(chain)]
            if [float(a["temperature"]) for a in chain] != expected_temperatures:
                raise ValueError(f"Fallback temperature sequence invalid: {item_key}")
            if any(a["accepted"] for a in chain[:-1]):
                raise ValueError(f"Fallback continued after acceptance: {item_key}")
            accepted = bool(chain[-1]["accepted"])
            exhausted = bool(chain[-1]["returned_as_exhausted_last_attempt"])
            if accepted == exhausted or (exhausted and len(chain) != 5):
                raise ValueError(f"Fallback terminal status invalid: {item_key}")
        else:
            accepted = exhausted = False
        final = chain[-1] if chain else row
        final_rows.append({
            "session": item_key[0], "slot": item_key[1], "segment_index": item_key[2],
            "fallback_triggered": triggered, "fallback_accepted": accepted,
            "fallback_exhausted": exhausted, "fallback_attempt_count": len(chain),
            "final_generated_step_count": int(final["generated_step_count"]),
            "final_terminated_by_eos": bool(final["terminated_by_eos"]),
            "final_budget_touched": int(final["generated_step_count"]) == 224,
            "final_cap_hit_without_eos": (int(final["generated_step_count"]) == 224
                                           and not final["terminated_by_eos"]),
        })
    triggers = sum(r["fallback_triggered"] for r in final_rows)
    accepted = sum(r["fallback_accepted"] for r in final_rows)
    exhausted = sum(r["fallback_exhausted"] for r in final_rows)
    elapsed = None
    runtime_note = "Historical timing covers all AISHELL-5/source arms; subset timing unavailable."
    if not baseline:
        calls_path = directory / "generation_calls.jsonl"
        if calls_path.exists():
            calls = read_jsonl(calls_path)
            if calls and all("elapsed_seconds" in r for r in calls):
                elapsed = sum(float(r["elapsed_seconds"]) for r in calls)
                runtime_note = ("Sum of generation call elapsed_seconds in this C run; "
                                "generation and per-call validation (constrained score audit, "
                                "text decoding and token checks), excluding waveform/feature "
                                "preparation and cpCER scoring; "
                                "A subset timing unavailable.")
        if elapsed is None:
            runtime_note = "Generation timing unavailable in supplied logs."
    return {
        "source_directory": str(directory.resolve()), "windows": len(final_rows),
        "compression_triggered_windows": triggers,
        "fallback_accepted_windows": accepted, "fallback_exhausted_windows": exhausted,
        "fallback_acceptance_rate_among_triggered": accepted / triggers if triggers else None,
        "fallback_exhaustion_rate_among_triggered": exhausted / triggers if triggers else None,
        "fallback_active_window_attempts": len(attempts),
        "t0_budget_touched_windows": sum(int(r["generated_step_count"]) == 224 for r in t0_rows),
        "fallback_budget_touched_attempts": sum(int(r["generated_step_count"]) == 224 for r in attempts),
        "final_budget_touched_windows": sum(r["final_budget_touched"] for r in final_rows),
        "final_cap_hit_without_eos_windows": sum(r["final_cap_hit_without_eos"] for r in final_rows),
        "generation_elapsed_seconds": elapsed, "runtime_note": runtime_note,
        "runtime_comparison_with_A_available": False,
        "probability_semantics": ("legacy generated.scores temperature removal and renormalization; "
                                  + ("n=0 distribution" if baseline else "n=4 constrained distribution")),
    }, final_rows


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def dump_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def render_report(summary: dict) -> str:
    lines = [
        "TMM n-gram blocking post-primary v1: Primary/public",
        f"Status: {summary['status']}",
        "Scope: AISHELL-5 Primary 18 full meetings; public Sortformer; 1744 stream windows; 222 original batches.",
        "All empty-output windows retained. Each arm gets one optimal speaker assignment per full meeting.",
        "A = cap224 + compression-triggered fallback; B = A + canonical guard.",
        "C = same decoder with token n=4 on T0 and every fallback; D = C + same guard.",
        "", "Arm | micro cpCER | S / D / I | S/N / D/N / I/N | residual events/windows | raw chars / ratio to A | normalized chars / ratio to A | raw/normalized empty windows | guard removed raw/normalized",
    ]
    for arm in "ABCD":
        row = summary["arms"][arm]
        if row["status"] != "available":
            lines.append(f"{arm} | UNAVAILABLE: C/D have not been decoded; no estimate or proxy is substituted.")
            continue
        lines.append(
            f"{arm} | {row['micro_cpcer']:.9f} ({100 * row['micro_cpcer']:.4f}%) | "
            f"{row['substitutions']} / {row['deletions']} / {row['insertions']} | "
            f"{row['substitution_rate']:.9f} / {row['deletion_rate']:.9f} / {row['insertion_rate']:.9f} | "
            f"{row['residual_canonical_guard_events']}/{row['residual_canonical_guard_windows']} | "
            f"{row['raw_characters']} / {row['raw_length_ratio_to_A']:.6f} | "
            f"{row['normalized_characters']} / {row['normalized_length_ratio_to_A']:.6f} | "
            f"{row['raw_empty_windows']}/{row['normalized_empty_windows']} | "
            f"{row['guard_removed_raw_characters']}/{row['guard_removed_normalized_characters']}"
        )
    lines += ["", f"Reference characters N = {summary['reference_characters']}; all-empty hypothesis cpCER = 1.0.",
              "Raw characters: Python len(final per-window text), including spaces/punctuation, before normalization.",
              "Normalized characters: sum(len(normalize_zh_text(final per-window text))); NFKC, casefold, annotation removal, Unicode letters/numbers only.",
              "Guard is applied to raw per-window text with (P,R,S,K)=(12,6,24,2); deletion is relative to that arm's pre-guard parent.",
              "Residual events count the ordered canonical fixed-point edit trace, not overlapping candidate spans.",
              "", "Prespecified comparisons (negative means the first arm is better):"]
    for label, row in summary["comparisons"].items():
        if row["status"] != "available":
            lines.append(f"{label}: unavailable (C/D missing).")
        else:
            lines.append(f"{label}: delta={row['delta_micro_cpcer']:.9f} "
                         f"({row['delta_percentage_points']:+.4f} percentage points); "
                         f"95% CI [{row['ci95_lower']:.9f}, {row['ci95_upper']:.9f}]; "
                         f"W/T/L={row['wins']}/{row['ties']}/{row['losses']}.")
    lines += ["Bootstrap: 50000 paired meeting resamples; seed 20261002; recompute sum(errors)/sum(N) in each draw.",
              "", "Decode diagnostics (active window attempts; inactive batch members are excluded from attempt counts):"]
    for arm, diag in summary["decode_diagnostics"].items():
        lines.append(f"{arm}: triggers={diag['compression_triggered_windows']}; "
                     f"accepted={diag['fallback_accepted_windows']}; exhausted={diag['fallback_exhausted_windows']}; "
                     f"attempts={diag['fallback_active_window_attempts']}; "
                     f"224-token budget touched T0/fallback attempts/final="
                     f"{diag['t0_budget_touched_windows']}/{diag['fallback_budget_touched_attempts']}/{diag['final_budget_touched_windows']}; "
                     f"final cap without EOS={diag['final_cap_hit_without_eos_windows']}.")
        lines.append(f"{arm} timing: {diag['generation_elapsed_seconds']} seconds. {diag['runtime_note']}")
    lines += ["Budget touched counts include a valid EOS at position 224; without-EOS counts are reported separately.",
              "Fallback threshold -1 retains legacy processed-score semantics; under n=4 it reflects the constrained distribution.",
              "", "Validation: A/B texts, full-meeting S/D/I counts and assignments match V5 exactly.",
              "B/D zero residual events follows from guard fixed-point behavior; it is not evidence of semantic preservation.",
              "This is an additional post-primary comparison restricted to Primary/public; old listening annotations do not estimate C/D semantic safety."]
    if summary["status"] != "complete":
        lines += ["", "C/D and all blocking comparisons remain pending. This A/B report does not complete the planned experiment."]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c-dir", type=Path, help="Complete C decode output directory; omitted = A/B only")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT / "baseline_analysis")
    parser.add_argument("--v4-dir", type=Path, default=V4)
    parser.add_argument("--v5-dir", type=Path, default=V5)
    parser.add_argument("--split-manifest", type=Path,
                        default=PROJECT / "manifests/streaming_v035_aishell5_eval36_fullmeeting_v1.jsonl")
    parser.add_argument("--references", type=Path,
                        default=PROJECT / "experiments/streaming_v035_aishell5_eval36_fullmeeting_v1/meeting_full_references.jsonl")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    for path in (args.v4_dir, args.v5_dir, args.output_dir, args.references, args.split_manifest):
        legacy_score.assert_not_test(path)
    output = args.output_dir.resolve()
    protected = (args.v4_dir.resolve(), args.v5_dir.resolve())
    if any(output == p or p in output.parents for p in protected):
        raise ValueError("Output directory must not be inside original V4/V5 results")
    manifests = read_jsonl(args.split_manifest)
    primary = [r for r in manifests if r.get("analysis_role") == "primary"]
    sessions = {r["session"] for r in primary}
    if len(primary) != 18 or len(sessions) != 18:
        raise ValueError("Primary manifest must identify exactly 18 unique full meetings")
    if any(r.get("full_meeting") is not True for r in primary):
        raise ValueError("Primary selection requires full-meeting manifest rows")
    selected_reference_rows = [r for r in read_jsonl(args.references) if r["session"] in sessions]
    if {r["session"] for r in selected_reference_rows} != sessions:
        raise ValueError("Missing Primary full-meeting references")
    a_rows = select_rows(read_jsonl(args.v4_dir / "cap224_fallback_hypotheses.jsonl"), sessions, SOURCE_SYSTEM)
    if len(a_rows) != 1744:
        raise ValueError(f"Expected 1744 Primary/public windows, got {len(a_rows)}")
    audio = select_rows(read_jsonl(args.v4_dir / "all_audio_manifest.jsonl"), sessions, SOURCE_SYSTEM)
    validate_audio_coverage(audio, a_rows)
    batches = select_rows(read_jsonl(args.v4_dir / "decode_batch_map.jsonl"), sessions, SOURCE_SYSTEM)
    if set(index_rows(batches, "batch map")) != set(index_rows(a_rows, "A")):
        raise ValueError("Original batch map does not cover the 1744 selected windows")
    batch_count = len({r["batch_id"] for r in batches})
    if batch_count != 222:
        raise ValueError(f"Expected 222 original batches, got {batch_count}")
    arms = {"A": [dict(row, system="A") for row in a_rows]}
    arms["B"], guard_traces = guarded_rows(a_rows, "B")
    old_rows = read_jsonl(args.v5_dir / "control_hypotheses_fourteen_systems.jsonl")
    for arm, system in V5_SYSTEMS.items():
        old = select_rows(old_rows, sessions, system)
        assert_same_windows(arms[arm], old, f"V5 {arm}", compare_text=True)
    c_verification = {"status": "unavailable", "reason": "C/D have not been decoded"}
    if args.c_dir is not None:
        legacy_score.assert_not_test(args.c_dir)
        c_verification = validate_c_run(args.c_dir)
        c_rows = select_rows(read_jsonl(args.c_dir / "cap224_ngram_fallback_hypotheses.jsonl"),
                             sessions, SOURCE_SYSTEM)
        assert_same_windows(a_rows, c_rows, "C")
        arms["C"] = [dict(row, system="C") for row in c_rows]
        arms["D"], d_traces = guarded_rows(c_rows, "D")
        guard_traces.extend(d_traces)
    stats, details = score_arms(arms, selected_reference_rows)
    v5_verification = verify_v5_scores(details, args.v5_dir / "control_cpcer_fourteen_systems.csv")
    baseline_raw = sum(len(r["text"]) for r in a_rows)
    baseline_normalized = sum(len(legacy_score.normalize_zh_text(r["text"])) for r in a_rows)
    residual_traces = []
    for arm, rows in arms.items():
        source = arms[{"B": "A", "D": "C"}.get(arm, arm)]
        metrics, traces = morphology(rows, source, baseline_raw, baseline_normalized)
        stats[arm].update(metrics)
        residual_traces.extend(traces)
    for arm in "ABCD":
        stats.setdefault(arm, {"status": "unavailable", "reason": "C/D have not been decoded"})
    diag_a, diag_rows_a = decode_diagnostics(args.v4_dir, sessions, a_rows, baseline=True)
    diagnostics, diagnostic_rows = {"A_B": diag_a}, [dict(r, decoder_arm="A_B") for r in diag_rows_a]
    if args.c_dir is not None:
        diag_c, diag_rows_c = decode_diagnostics(args.c_dir, sessions, arms["C"], baseline=False)
        diagnostics["C_D"] = diag_c
        diagnostic_rows.extend(dict(r, decoder_arm="C_D") for r in diag_rows_c)
    summary = {
        "schema": "tmm_ngram_blocking_postprimary_analysis_v1",
        "status": "complete" if args.c_dir is not None else "partial_A_B_only",
        "scope": "AISHELL-5 Primary/public full meetings", "source_system": SOURCE_SYSTEM,
        "sessions": sorted(sessions), "meetings": 18, "windows_per_arm": 1744,
        "original_batches": batch_count, "reference_characters": stats["A"]["reference_chars"],
        "normalization": legacy_score.NORMALIZATION_ID,
        "scorer": "unchanged scripts/score_meeting_cpcer.py and realmeetsep.cpcer",
        "levenshtein_backend": levenshtein_backend_info(),
        "permutation_scope": "one optimal assignment per arm per complete recording",
        "empty_hypothesis_cpcer": 1.0, "empty_output_windows_retained": True,
        "guard_parameters": GUARD_PARAMS,
        "bootstrap": {"repetitions": 50000, "seed": 20261002, "unit": "paired full meeting",
                      "estimator": "sum(errors) / sum(reference_characters)", "interval": "percentile 95%"},
        "character_count_definitions": {
            "raw": "sum Python len(final per-window text), including punctuation/spaces",
            "normalized": "sum len(normalize_zh_text(final per-window text))",
            "empty_raw": "final text == empty string", "empty_normalized": "normalized final text == empty string",
            "guard_deletions": "parent arm minus guarded final arm; raw and normalized both reported",
            "residual_events": "ordered canonical raw-text guard edit trace at fixed point; overlapping candidates not counted",
        },
        "arms": stats, "comparisons": paired_bootstrap(details),
        "decode_diagnostics": diagnostics, "v5_verification": v5_verification,
        "c_run_verification": c_verification,
        "sources": {"v4_directory": str(args.v4_dir.resolve()), "v5_directory": str(args.v5_dir.resolve()),
                    "c_directory": str(args.c_dir.resolve()) if args.c_dir else None,
                    "references": str(args.references.resolve()), "split_manifest": str(args.split_manifest.resolve())},
    }
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "analysis.json", summary)
    dump_jsonl(output / "hypotheses.jsonl", [r for rows in arms.values() for r in rows])
    dump_jsonl(output / "references.jsonl", selected_reference_rows)
    dump_jsonl(output / "guard_edit_traces.jsonl", guard_traces)
    dump_jsonl(output / "residual_guard_traces.jsonl", residual_traces)
    dump_jsonl(output / "per_window_decode_diagnostics.jsonl", diagnostic_rows)
    dump_jsonl(output / "per_meeting.jsonl", details)
    dump_csv(output / "per_meeting.csv", details)
    dump_csv(output / "arm_summary.csv", [dict(arm=arm, **stats[arm]) for arm in "ABCD"])
    dump_csv(output / "paired_comparisons.csv", [dict(comparison=k, **v) for k, v in summary["comparisons"].items()])
    (output / "report.txt").write_text(render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_directory": str(args.output_dir.resolve()),
                      "available_arms": [a for a, v in summary["arms"].items() if v["status"] == "available"],
                      "v5_verification": summary["v5_verification"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
