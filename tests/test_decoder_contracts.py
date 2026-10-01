"""Optional synthetic tensor tests: no model downloads, corpus, or GPU required."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from realmeetsep.asr_whisper_fallback_v039 import openai_average_logprob_from_scaled_scores, openai_average_logprob_batch_from_scaled_scores


@unittest.skipUnless(HAS_TORCH, "optional decoder dependencies are not installed")
class DecoderContracts(unittest.TestCase):
    def test_temperature_eos_and_no_eos_denominators(self):
        base = [torch.tensor([0., 2., 1.]), torch.tensor([0., 1., 3.])]
        for temperature in (0., .2, 1.):
            scaled = [x / (temperature or 1.) for x in base]
            result = openai_average_logprob_from_scaled_scores([10, 11, 1, 2], scaled, temperature=temperature, eos_token_id=2, expected_prompt_ids=[10, 11])
            expected = sum(float(torch.log_softmax(x, dim=0)[token]) for x, token in zip(base, (1, 2)))
            self.assertEqual(result.denominator, 2)
            self.assertAlmostEqual(result.sequence_logprob_sum, expected, places=6)
            unfinished = openai_average_logprob_from_scaled_scores([10, 11, 1, 2], scaled, temperature=temperature, eos_token_id=99, expected_prompt_ids=[10, 11])
            self.assertEqual(unfinished.denominator, 3)
            self.assertAlmostEqual(unfinished.avg_logprob_openai, expected / 3, places=6)

    def test_batch_single_equivalence_and_padding(self):
        scores = [torch.tensor([[0., 2., 1.], [0., 1., 2.]])] * 3
        sequences = [[10, 11, 1, 2, 2], [10, 11, 1, 1, 1]]
        batch = openai_average_logprob_batch_from_scaled_scores(sequences, scores, temperature=.4, eos_token_id=2, expected_prompt_ids=[10, 11])
        for index in range(2):
            single = openai_average_logprob_from_scaled_scores(sequences[index], [x[index] for x in scores], temperature=.4, eos_token_id=2, expected_prompt_ids=[10, 11])
            self.assertEqual(batch[index].denominator, single.denominator)
            self.assertAlmostEqual(batch[index].avg_logprob_openai, single.avg_logprob_openai, places=6)
        self.assertEqual(batch[0].generated_step_count, 2)
        self.assertEqual(batch[1].denominator, 4)

    def test_prompt_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            openai_average_logprob_from_scaled_scores([99, 1], [torch.tensor([0., 1.])], temperature=1., eos_token_id=1, expected_prompt_ids=[10])
