from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from realmeetsep.asr_whisper_fallback_v039 import openai_average_logprob_from_scaled_scores
from realmeetsep.asr_whisper_ngram_postprimary_v1 import constrained_average_logprob_batch, validate_token_history

import decode_whisper_ngram_postprimary_v1 as decoder

PROMPT = [10, 11, 12, 13]
EOS = 5
CHECK = unittest.TestCase()


def test_scores_ignore_only_post_eos_padding_and_match_single_oracle():
    sequences = [[*PROMPT, 1, EOS, EOS], [*PROMPT, 2, 3, 4]]
    scores = [torch.arange(12, dtype=torch.float32).reshape(2, 6) / 7 for _ in range(3)]
    scores[2][0, EOS] = -torch.inf
    observed = constrained_average_logprob_batch(sequences, scores, temperature=0.4, eos_token_id=EOS, expected_prompt_ids=PROMPT)
    for row, value in enumerate(observed):
        expected = openai_average_logprob_from_scaled_scores(sequences[row], [score[row] for score in scores], temperature=0.4, eos_token_id=EOS, expected_prompt_ids=PROMPT)
        CHECK.assertAlmostEqual(value.avg_logprob_openai, expected.avg_logprob_openai, delta=2e-7)
        assert value.denominator == expected.denominator
    assert observed[0].denominator == 2
    assert observed[1].denominator == 4
    scores[1][0, EOS] = -torch.inf
    with CHECK.assertRaisesRegex(ValueError, "non-finite"):
        constrained_average_logprob_batch(sequences, scores, temperature=0.4, eos_token_id=EOS, expected_prompt_ids=PROMPT)


def audit(sequence, **kwargs):
    options = dict(prompt_ids=PROMPT, score_count=len(sequence) - 4, eos_token_id=EOS, pad_token_id=EOS,
                   ngram_size=4, max_new_tokens=224, timestamp_begin=100)
    options.update(kwargs)
    return validate_token_history(sequence, **options)


def test_ngram_history_includes_prompt_and_excludes_eos_padding():
    with CHECK.assertRaisesRegex(ValueError, "repeated 4-gram"):
        audit([*PROMPT, *PROMPT, EOS])
    value = audit([*PROMPT, 1, EOS, EOS, EOS, EOS, EOS, EOS])
    assert value["valid_generated_token_ids"] == [1, EOS]
    assert value["score_step_count"] == 7
    assert value["ngram_constraint_verified"]


def test_history_checks_budget_prompt_and_records_timestamp_without_new_suppression():
    with CHECK.assertRaisesRegex(ValueError, "four-token"):
        audit([99, 11, 12, 13, EOS])
    with CHECK.assertRaisesRegex(ValueError, "before reaching"):
        audit([*PROMPT, 1, 2])
    assert audit([*PROMPT, 111, EOS])["generated_timestamp_token"]
    with CHECK.assertRaisesRegex(ValueError, "budget"):
        audit([*PROMPT, *range(200, 425)])


def fake_maps():
    rows = []
    for batch in range(3):
        for position in range(2):
            rows.append(dict(system=decoder.PUBLIC_SYSTEM, session=f"meeting{batch}", slot=0,
                segment_index=position, batch_id=f"b{batch}", group_id=f"g{batch}",
                group_order=batch, source_row_index=position, batch_position=position,
                actual_batch_size=2, corpus="aishell5"))
    return rows


def test_subsetting_retains_original_indices_positions_and_full_batches():
    rows = fake_maps()
    batches = decoder.build_original_batches(rows[2:4], rows)
    assert [batch.global_batch_index for batch in batches] == [1]
    assert [row["segment_index"] for row in batches[0].rows] == [0, 1]
    assert decoder.sampling_seed(1, batches[0].global_batch_index) == 20360726
    with CHECK.assertRaisesRegex(ValueError, "splits original batch"):
        decoder.build_original_batches(rows[2:3], rows)
    with CHECK.assertRaisesRegex(ValueError, "original map record order"):
        decoder.build_original_batches(list(reversed(rows[2:4])), rows)


def test_common_ngram_parameter_and_protocol_rejects_old_zero_override():
    for temperature in (0.0, *decoder.TEMPERATURES):
        params = decoder.generation_parameters(4, temperature)
        assert params["no_repeat_ngram_size"] == 4
        assert params["max_new_tokens"] == 224
    protocol = {"ngram_size": 4, "max_new_tokens": 224, "sampling_seed": 20260725,
                "fallback": {"sampling_controls": {"no_repeat_ngram_size": 0}}}
    with CHECK.assertRaisesRegex(ValueError, "sampling controls"):
        decoder.validate_protocol(protocol)


def test_fallback_keeps_inactive_full_batch_members_and_uses_original_seed():
    with TemporaryDirectory() as directory:
        check_fallback(Path(directory))


def test_exhausted_fallback_returns_final_attempt():
    with TemporaryDirectory() as directory:
        check_fallback(Path(directory), exhaust=True)


def check_fallback(tmp_path, exhaust=False):
    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)
    rows = []
    for i in range(2):
        rows.append(dict(system=decoder.PUBLIC_SYSTEM, session="meeting", slot=i, segment_index=0,
                         audio_filepath=str(path), duration=0.1, start=0.0, end=0.1, meeting_stream=f"slot{i}"))
    class Processor:
        count = 0
        def __call__(self, arrays, **kwargs):
            assert len(arrays) == 2
            return SimpleNamespace(input_features=torch.zeros(2, 1), attention_mask=torch.ones(2, 1))
        def batch_decode(self, sequences, **kwargs):
            if exhaust:
                return ["abcd" * 100, "ok"]
            result = [["abcd" * 100, "ok"], ["abcd" * 100, "discarded"], ["accepted", "discarded2"]][self.count]
            self.count += 1
            return result
    class Model:
        generation_config = SimpleNamespace(eos_token_id=EOS, pad_token_id=EOS, no_timestamps_token_id=99)
        calls = []
        def generate(self, features, **kwargs):
            self.calls.append((features.shape[0], kwargs, torch.initial_seed()))
            sequences = torch.tensor([[*PROMPT, 1, EOS], [*PROMPT, 2, EOS]])
            scores = [torch.zeros(2, 16), torch.zeros(2, 16)]
            scores[0][0, 1] = scores[0][1, 2] = 15
            scores[1][:, EOS] = 15
            return SimpleNamespace(sequences=sequences, scores=scores)
    model = Model()
    value = decoder.decode_batch(decoder.OriginalBatch("original_batch", 411, "group", tuple(rows)),
        processor=Processor(), model=model, device=torch.device("cpu"), dtype=torch.float32, prompt=PROMPT, ngram_size=4)
    if exhaust:
        assert len(model.calls) == 6
        assert all(size == 2 for size, _, _ in model.calls)
        assert len(value["attempts"]) == 5
        assert not any(row["accepted"] for row in value["attempts"])
        assert value["attempts"][-1]["returned_as_exhausted_last_attempt"]
        assert value["final_diagnostics"][0]["final_temperature"] == 1.0
        assert value["final_diagnostics"][0]["fallback_exhausted"]
        return
    assert [size for size, _, _ in model.calls] == [2, 2, 2]
    assert [seed for _, _, seed in model.calls[1:]] == [20261136, 20361136]
    assert value["fallback_hypotheses"][1]["text"] == "ok"
    assert value["fallback_hypotheses"][0]["text"] == "accepted"
    assert [call["active_positions"] for call in value["generation_calls"]] == [[0, 1], [0], [0]]
    assert len(value["attempts"]) == 2
    assert value["attempts"][-1]["accepted"]
    assert all(call["generation_parameters"]["no_repeat_ngram_size"] == 4 for call in value["generation_calls"])
    assert value["final_diagnostics"][0]["fallback_attempt_count"] == 2


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in list(globals().items()) if name.startswith("test_") and callable(value))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
