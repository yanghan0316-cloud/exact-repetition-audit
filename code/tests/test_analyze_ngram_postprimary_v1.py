from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import analyze_ngram_postprimary_v1 as analysis


def hypothesis(slot=0, segment_index=0, text="", system="A"):
    return {
        "schema": "meeting_stream_hypothesis_segment_v1", "session": "meeting",
        "split": "eval", "system": system, "asr_model": "toy@sha256:" + "a" * 64,
        "meeting_stream": f"slot{slot}", "slot": slot, "segment_index": segment_index,
        "start": segment_index * 30.0, "end": (segment_index + 1) * 30.0, "text": text,
    }


def reference(texts):
    return {
        "schema": "meeting_cpcer_reference_v1", "session": "meeting", "split": "eval",
        "metric_scope": "full_meeting", "character_time_alignment_used": False,
        "normalization": analysis.legacy_score.NORMALIZATION_ID,
        "speakers": list(texts), "references": texts,
    }


class NgramAnalysisTests(unittest.TestCase):
    def test_empty_windows_are_not_lost_and_empty_cpcer_is_one(self):
        rows = [hypothesis(segment_index=0), hypothesis(segment_index=1)]
        stats, detail = analysis.score_arms({"A": rows}, [reference({"speaker": "甲乙"})])
        self.assertEqual(stats["A"]["micro_cpcer"], 1.0)
        self.assertEqual(detail[0]["deletions"], 2)
        shape, _ = analysis.morphology(rows, rows, 0, 0)
        self.assertEqual(shape["window_count"], 2)
        self.assertEqual(shape["raw_empty_windows"], 2)

    def test_speaker_assignment_is_once_per_meeting_not_per_window(self):
        # Each window alone could be permuted to perfect agreement. No single
        # full-meeting assignment aligns both because streams swap mid-meeting.
        rows = [hypothesis(0, 0, "甲"), hypothesis(0, 1, "丁"),
                hypothesis(1, 0, "丙"), hypothesis(1, 1, "乙")]
        stats, _ = analysis.score_arms({"A": rows}, [reference({"s1": "甲乙", "s2": "丙丁"})])
        self.assertEqual(stats["A"]["errors"], 2)
        self.assertEqual(stats["A"]["micro_cpcer"], 0.5)

    def test_coverage_duplicates_and_changed_boundaries_rejected(self):
        row = hypothesis()
        with self.assertRaisesRegex(ValueError, "coverage"):
            analysis.assert_same_windows([row], [], "C")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.assert_same_windows([row], [row, row], "C")
        with self.assertRaisesRegex(ValueError, "end differs"):
            analysis.assert_same_windows([row], [dict(row, end=29)], "C")

    def test_guard_trace_counts_edits_and_raw_normalized_lengths(self):
        row = hypothesis(text="Ａ，" + "哈" * 30 + "！")
        guarded, traces = analysis.guarded_rows([row], "B")
        self.assertEqual(len(traces), 1)
        self.assertEqual(len(traces[0]["events"]), 1)
        self.assertEqual(guarded[0]["text"], "Ａ，哈哈！")
        stats, residual = analysis.morphology(guarded, [row], len(row["text"]), 31)
        self.assertEqual(stats["raw_characters"], 5)
        self.assertEqual(stats["normalized_characters"], 3)
        self.assertEqual(stats["guard_removed_raw_characters"], 28)
        self.assertEqual(stats["guard_removed_normalized_characters"], 28)
        self.assertEqual(stats["residual_canonical_guard_events"], 0)
        self.assertEqual(residual, [])

    def test_bootstrap_uses_resampled_micro_counts_and_common_seed(self):
        rows = []
        # Strongly unequal denominators make macro and micro clearly different.
        for arm, errors in {"A": [1, 20, 3], "B": [1, 19, 3],
                            "C": [0, 21, 3], "D": [0, 20, 3]}.items():
            for session, numerator, denominator in zip(["m1", "m2", "m3"], errors, [1, 100, 10]):
                rows.append({"arm": arm, "session": session, "errors": numerator,
                             "reference_chars": denominator})
        report = analysis.paired_bootstrap(rows, repetitions=50000, seed=20261002)
        comparison = report["C-A"]
        self.assertEqual(comparison["delta_micro_cpcer"], 0)
        self.assertEqual((comparison["wins"], comparison["ties"], comparison["losses"]), (1, 1, 1))
        indices = np.random.default_rng(20261002).integers(0, 3, size=(50000, 3))
        expected = np.array([-1, 1, 0])[indices].sum(axis=1) / np.array([1, 100, 10])[indices].sum(axis=1)
        self.assertEqual(comparison["ci95_lower"], float(np.percentile(expected, 2.5)))
        self.assertEqual(comparison["ci95_upper"], float(np.percentile(expected, 97.5)))
        self.assertNotAlmostEqual(float(expected.mean()), float(np.mean([-1, .01, 0])))

    def test_missing_c_is_explicit_and_incomplete_pairs_rejected(self):
        rows = [{"arm": "A", "session": "m1", "errors": 1, "reference_chars": 1},
                {"arm": "B", "session": "m1", "errors": 0, "reference_chars": 1}]
        report = analysis.paired_bootstrap(rows)
        self.assertTrue(all(r["status"] == "unavailable" for r in report.values()))
        rows.append({"arm": "C", "session": "m2", "errors": 0, "reference_chars": 1})
        with self.assertRaisesRegex(ValueError, "same meetings"):
            analysis.paired_bootstrap(rows)

    def test_fallback_acceptance_and_exhaustion_with_cap_eos_distinction(self):
        with tempfile.TemporaryDirectory(prefix="ngram-analysis-") as directory:
            root = Path(directory)
            expected = [hypothesis(segment_index=0), hypothesis(segment_index=1)]
            base = {"session": "meeting", "slot": 0, "system": analysis.SOURCE_SYSTEM}
            scores = [dict(base, segment_index=0, generated_step_count=224,
                           terminated_by_eos=False, compression_ratio_gt_2_4=True),
                      dict(base, segment_index=1, generated_step_count=224,
                           terminated_by_eos=True, compression_ratio_gt_2_4=False)]
            attempts = [dict(base, segment_index=0, generated_step_count=15,
                             terminated_by_eos=True, temperature=.2, accepted=True,
                             returned_as_exhausted_last_attempt=False)]
            analysis.dump_jsonl(root / "cap224_t0_scores.jsonl", scores)
            analysis.dump_jsonl(root / "temperature_fallback_attempts.jsonl", attempts)
            stats, details = analysis.decode_diagnostics(root, {"meeting"}, expected, baseline=True)
            self.assertEqual(stats["compression_triggered_windows"], 1)
            self.assertEqual(stats["fallback_accepted_windows"], 1)
            self.assertEqual(stats["final_budget_touched_windows"], 1)
            self.assertEqual(stats["final_cap_hit_without_eos_windows"], 0)
            self.assertEqual(details[0]["final_generated_step_count"], 15)
            attempts[0]["accepted"] = False
            analysis.dump_jsonl(root / "temperature_fallback_attempts.jsonl", attempts)
            with self.assertRaisesRegex(ValueError, "terminal status"):
                analysis.decode_diagnostics(root, {"meeting"}, expected, baseline=True)

    def test_c_rejects_smoke_or_wrong_ngram_as_formal_experiment(self):
        with tempfile.TemporaryDirectory(prefix="ngram-analysis-") as directory:
            root = Path(directory)
            summary = {"ngram_size": 4, "max_new_tokens": 224, "partial_smoke": False,
                       "decoded_records": 1744, "decoded_batches": 222, "decoded_sessions": 18}
            analysis.dump_json(root / "decoder_summary.json", summary)
            self.assertEqual(analysis.validate_c_run(root)["status"], "passed")
            for field, value in (("ngram_size", 0), ("partial_smoke", True), ("decoded_records", 16)):
                analysis.dump_json(root / "decoder_summary.json", dict(summary, **{field: value}))
                with self.assertRaisesRegex(ValueError, field):
                    analysis.validate_c_run(root)


if __name__ == "__main__":
    unittest.main()
