"""Synthetic contracts; every string here is constructed, not a corpus excerpt."""
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
from realmeetsep.asr_repetition_guard_v035 import guard_repetitions
from realmeetsep.cpcer import StreamTranscript, normalize_zh_text, score_cpcer, levenshtein_counts, levenshtein_counts_python


class GuardContracts(unittest.TestCase):
    def test_frozen_implementation_validation(self):
        for kwargs in ({"max_period_chars": 0}, {"min_repetitions": 2}, {"min_span_chars": 0}, {"keep_repetitions": 0}):
            with self.assertRaises(ValueError):
                guard_repetitions("abc", **kwargs)

    def test_span_gate_and_repetition_gate(self):
        self.assertEqual(guard_repetitions("abc" * 7).text, "abc" * 7)
        self.assertEqual(guard_repetitions("abc" * 8).text, "abc" * 2)
        self.assertEqual(guard_repetitions("abcde" * 5).text, "abcde" * 5)

    def test_largest_then_shortest_then_earliest(self):
        result = guard_repetitions("abcd" * 6 + "!" + "xyz" * 20)
        self.assertEqual((result.events[0].start, result.events[0].period_chars), (25, 3))
        same = guard_repetitions("ab" * 12 + "!" + "cd" * 12)
        self.assertEqual((same.events[0].start, same.events[0].period_chars), (0, 2))

    def test_edit_trace_reconstructs_original(self):
        original = "abcd" * 6 + "!" + "xyz" * 20
        result = guard_repetitions(original)
        reconstructed = result.text
        for event in reversed(result.events):
            reconstructed = reconstructed[:event.start] + event.unit * event.repetitions + reconstructed[event.start + event.kept_span_chars:]
        self.assertEqual(reconstructed, original)

    def test_generated_fixed_point_and_subsequence(self):
        for unit in ("a", "ab", "abc", "甲乙丙丁", "😀x"):
            for count in (0, 5, 6, 8, 24):
                text = "LEFT!" + unit * count + "!RIGHT"
                first = guard_repetitions(text)
                self.assertFalse(guard_repetitions(first.text).events)
                cursor = iter(text)
                self.assertTrue(all(any(c == wanted for c in cursor) for wanted in first.text))
                self.assertEqual(first.removed_characters, sum(e.original_span_chars - e.kept_span_chars for e in first.events))


class ScoreContracts(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_zh_text("Ａb，１２ <noise> [x] {y} 中！"), "ab12中")

    def test_assignment_padding_and_empty_anchor(self):
        refs = {"speaker_a": "abc", "speaker_b": "xyz"}
        score = score_cpcer(refs, [StreamTranscript("slot0", "xyz"), StreamTranscript("slot1", "abc")])
        self.assertEqual(score.counts.errors, 0)
        self.assertEqual(dict(score.assignment), {"speaker_a": "slot1", "speaker_b": "slot0"})
        empty = score_cpcer(refs, [])
        self.assertEqual(empty.counts.cer, 1.0)
        self.assertEqual(empty.counts.deletions, 6)
        self.assertEqual(score_cpcer({"a": "ab"}, [StreamTranscript("0", "abcabcabc")]).counts.cer, 3.5)

    def test_distance_backends_agree_on_total_error(self):
        words = [""] + ["".join(x) for n in range(1, 4) for x in itertools.product("ab", repeat=n)]
        for ref in words:
            for hyp in words:
                self.assertEqual(levenshtein_counts(ref, hyp).errors, levenshtein_counts_python(ref, hyp).errors)


def load_tests(loader, standard_tests, pattern):
    # Run the original dependency-free pytest-style tests using unittest.
    for filename in ("frozen_guard_cases.py", "frozen_baseline_cases.py"):
        spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), ROOT / "tests" / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name in sorted(vars(module)):
            value = getattr(module, name)
            if name.startswith("test_") and callable(value):
                standard_tests.addTest(unittest.FunctionTestCase(value))
    return standard_tests
