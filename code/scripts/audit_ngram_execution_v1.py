#!/usr/bin/env python3
"""Independently audit n-gram experiment logs without running model inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import zlib

PROJECT = Path(__file__).resolve().parents[1]
TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0]
PUBLIC = "raw_public_sortformer_v2_low_packet_gated"
PROMPT = [50258, 50260, 50359, 50363]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def key(row):
    return row["system"], row["session"], row["slot"], row["segment_index"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)


def expected_parameters(temperature):
    value = dict(language="zh", task="transcribe", condition_on_prev_tokens=False,
                 return_timestamps=False, force_unique_generate_call=True,
                 num_beams=1, num_return_sequences=1, max_new_tokens=224,
                 return_dict_in_generate=True, output_scores=True,
                 do_sample=temperature > 0, no_repeat_ngram_size=4)
    if temperature > 0:
        value.update(temperature=temperature, top_k=0, top_p=1.0, typical_p=1.0,
                     min_p=None, epsilon_cutoff=0.0, eta_cutoff=0.0, repetition_penalty=1.0)
    return value


def score_boundary_probe():
    """Check score arithmetic against independent log-softmax through EOS."""
    import torch
    sys.path.insert(0, str(PROJECT / "src"))
    from realmeetsep.asr_whisper_ngram_postprimary_v1 import constrained_average_logprob_batch
    prompt = [2, 3, 4, 5]
    eos = 9
    sequences = [prompt + [1, 1, eos] + [eos] * 221,
                 prompt + [1] * 224,
                 prompt + [1] * 223 + [eos]]
    scores = []
    for step in range(224):
        value = torch.tensor([[0.0, 1.0, -0.5, 0.25, 0.5, -1.0, 0.2, 0.8, -2.0, -0.1]] * 3)
        if step > 2:
            value[0, eos] = -float("inf")
        scores.append(value)
    output = constrained_average_logprob_batch(sequences, scores, temperature=0.4,
                 eos_token_id=eos, expected_prompt_ids=prompt)
    for row, stop in enumerate((3, 224, 224)):
        terms = []
        for step in range(stop):
            logits = [float(v) * 0.4 for v in scores[step][row]]
            token = sequences[row][4 + step]
            terms.append(logits[token] - math.log(sum(math.exp(v) for v in logits)))
        denominator = stop + (row == 1)
        require(math.isclose(output[row].avg_logprob_openai, sum(terms) / denominator, abs_tol=1e-6), "score arithmetic differs from independent log-softmax")
        require(output[row].denominator == denominator, "EOS/budget score denominator differs")
    bad_scores = [tensor.clone() for tensor in scores]
    bad_scores[1][0, 1] = -float("inf")
    try:
        constrained_average_logprob_batch(sequences, bad_scores, temperature=0.4,
                                         eos_token_id=eos, expected_prompt_ids=prompt)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite valid-position score was accepted")
    return {"early_eos_padding_negative_infinity_ignored": True,
            "first_eos_included": True, "no_eos_224_steps_denominator_225": True,
            "eos_on_224th_step_denominator_224": True,
            "nonfinite_valid_position_rejected": True}


def audit(run_dir: Path, require_complete: bool):
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "decoder_summary.json").read_text(encoding="utf-8"))
    protocol = metadata["protocol"]
    require(protocol["ngram_size"] == 4 and protocol["max_new_tokens"] == 224 and protocol["sampling_seed"] == 20260725, "protocol parameters differ")
    require(metadata["prompt_ids"] == PROMPT, "forced prompt differs")
    from transformers import GenerationConfig, WhisperTokenizer
    expected_config = GenerationConfig.from_pretrained(str(PROJECT / "models/whisper-base"), local_files_only=True).to_dict()
    expected_config["no_repeat_ngram_size"] = 4
    require(metadata["generation_config"] == expected_config, "base generation config differs beyond n=4")
    tokenizer = WhisperTokenizer.from_pretrained(str(PROJECT / "models/whisper-base"), local_files_only=True)
    map_rows = read_jsonl(metadata["decode_batch_map"])
    original_batches = {}
    global_indices = {}
    for row in map_rows:
        batch_id = row["batch_id"]
        if batch_id not in original_batches:
            global_indices[batch_id] = len(original_batches)
            original_batches[batch_id] = []
        original_batches[batch_id].append(row)
    meetings = read_jsonl(PROJECT / protocol["scope"]["meeting_manifest"])
    primary = {row["session"] for row in meetings if row["analysis_role"] == "primary"}
    expected_all = [row for row in map_rows if row["session"] in primary and row["system"] == PUBLIC]
    selected_indices = metadata["original_global_batch_indices"]
    expected_rows = [row for row in expected_all if global_indices[row["batch_id"]] in selected_indices]
    require([key(row) for row in metadata["selected_input_rows"]] == [key(row) for row in expected_rows], "selected input/order differs from original full map")
    require(len(expected_all) == 1744, "full source window inventory differs")
    if require_complete:
        require(len(expected_rows) == 1744 and len(selected_indices) == 222, "formal run is incomplete")
    calls = read_jsonl(run_dir / "generation_calls.jsonl")
    grouped = defaultdict(list)
    member_counter = Counter()
    all_valid_tokens, all_member_texts = [], []
    for call in calls:
        batch_id = call["original_decode_batch_id"]
        original = original_batches[batch_id]
        global_index = global_indices[batch_id]
        grouped[global_index].append(call)
        require(call["original_decode_batch_global_index"] == global_index, "global index renumbered")
        require(call["prompt_ids"] == PROMPT, "call prompt changed")
        require([key(row) for row in call["members"]] == [key(row) for row in original], "original batch member/order changed")
        require(call["sampled_batch_size"] == len(original), "original batch truncated")
        temperature = call["temperature"]
        index = None if temperature == 0 else TEMPERATURES.index(temperature)
        seed = None if index is None else 20260725 + 100000 * index + global_index
        require(call["temperature_index"] == index and call["sampling_seed"] == seed, "call seed/temperature index differs")
        require(call["generation_parameters"] == expected_parameters(temperature), "effective call parameters differ")
        active = set(call["active_positions"])
        for position, member in enumerate(call["members"]):
            require(member["original_decode_batch_position"] == position and member["original_decode_batch_size"] == len(original), "member position/size differs")
            require(member["original_decode_batch_global_index"] == global_index and member["original_decode_batch_id"] == batch_id, "member global batch differs")
            require(member["active_member"] == (position in active), "active-member flag differs")
            require(member["temperature"] == temperature and member["temperature_index"] == index and member["sampling_seed"] == seed, "member seed/temperature differs")
            require(member["no_repeat_ngram_size"] == 4, "blocking missing")
            seq = member["sequence_token_ids"]
            require(seq[:4] == PROMPT, "member prompt differs")
            generated = seq[4:]
            require(len(generated) == member["score_step_count"] and 1 <= len(generated) <= 224, "score count/token budget differs")
            eos = 50257 in generated
            stop = generated.index(50257) + 1 if eos else len(generated)
            valid = generated[:stop]
            require(all(token == 50257 for token in generated[stop:]), "nonpadding token after first EOS")
            require(eos or stop == 224, "no EOS before budget exhaustion")
            require(member["valid_generated_token_ids"] == valid and member["generated_step_count"] == stop, "valid-token inventory differs")
            require(member["terminated_by_eos"] == eos and member["cap_hit"] == (stop == 224) and member["no_eos_at_cap"] == (not eos), "EOS/cap flags differ")
            history = PROMPT + valid
            grams = [tuple(history[i:i + 4]) for i in range(len(history) - 3)]
            require(len(grams) == len(set(grams)), "repeated 4-gram in valid prompt-inclusive history")
            denominator = stop if eos else stop + 1
            require(member["denominator"] == denominator, "score denominator differs")
            require(math.isfinite(member["sequence_logprob_sum"]) and math.isfinite(member["avg_logprob_openai"]), "nonfinite valid score")
            require(close(member["avg_logprob_openai"], member["sequence_logprob_sum"] / denominator), "average logprob arithmetic differs")
            require((member["eos_logprob"] is not None and math.isfinite(member["eos_logprob"])) if eos else member["eos_logprob"] is None, "EOS score mismatch")
            text = member["text"]
            raw = text.encode("utf-8")
            ratio = len(raw) / len(zlib.compress(raw)) if raw else 0.0
            require(close(member["compression_ratio"], ratio), "compression ratio differs")
            require(member["generated_timestamp_token"] == any(token >= 50364 for token in valid), "timestamp diagnostic differs")
            member_counter["rows"] += 1
            member_counter["cap_hit"] += stop == 224
            member_counter["padding_rows"] += stop < len(generated)
            member_counter["timestamp_rows"] += any(token >= 50364 for token in valid)
            all_valid_tokens.append(seq)
            all_member_texts.append(text)
    require(tokenizer.batch_decode(all_valid_tokens, skip_special_tokens=True) == all_member_texts, "text differs from stored token IDs")
    require(list(grouped) == selected_indices, "generation batch inventory differs")
    t0_expected, attempt_expected, final_expected, diagnostics_expected = [], [], [], []
    for global_index, sequence in grouped.items():
        initial = sequence[0]
        require(initial["temperature"] == 0 and initial["active_positions"] == list(range(len(initial["members"]))), "initial call differs")
        active = {i for i, row in enumerate(initial["members"]) if row["compression_ratio"] > 2.4}
        triggered = set(active)
        final = list(initial["members"])
        histories = defaultdict(list)
        for row in initial["members"]:
            t0_expected.append(dict(row, triggered_fallback=row["original_decode_batch_position"] in triggered))
        for index, call in enumerate(sequence[1:]):
            require(active and index < 5, "unnecessary extra fallback call")
            require(call["temperature"] == TEMPERATURES[index] and call["active_positions"] == sorted(active), "active fallback trajectory differs")
            remaining = set()
            for position in sorted(active):
                row = call["members"][position]
                cr_pass, lp_pass = row["compression_ratio"] <= 2.4, row["avg_logprob_openai"] >= -1.0
                accepted = cr_pass and lp_pass
                exhausted = index == 4 and not accepted
                attempt = dict(row, compression_ratio_pass=cr_pass, avg_logprob_openai_pass=lp_pass,
                               accepted=accepted, returned_as_exhausted_last_attempt=exhausted,
                               inactive_original_batch_members_retained=len(call["members"]) - len(active))
                attempt_expected.append(attempt)
                histories[position].append(attempt)
                if accepted or exhausted:
                    final[position] = row
                else:
                    remaining.add(position)
            active = remaining
        require(not active, "unfinished fallback chain")
        final_expected.extend(final)
        for position, row in enumerate(final):
            diagnostics_expected.append({"key": key(row), "final_generated_step_count": row["generated_step_count"],
                "final_temperature": row["temperature"], "cap_hit": row["cap_hit"], "terminated_by_eos": row["terminated_by_eos"],
                "fallback_triggered": position in triggered, "fallback_accepted": any(r["accepted"] for r in histories[position]),
                "fallback_exhausted": any(r["returned_as_exhausted_last_attempt"] for r in histories[position]),
                "fallback_attempt_count": len(histories[position])})
    require(read_jsonl(run_dir / "t0_scores.jsonl") == t0_expected, "T0 score sidecar differs")
    require(read_jsonl(run_dir / "temperature_fallback_attempts.jsonl") == attempt_expected, "active attempts sidecar differs")
    for filename, expected in [("cap224_ngram_t0_hypotheses.jsonl", t0_expected), ("cap224_ngram_fallback_hypotheses.jsonl", final_expected)]:
        actual = read_jsonl(run_dir / filename)
        require([(key(r), r["text"]) for r in actual] == [(key(r), r["text"]) for r in expected], "final text/adoption differs: " + filename)
    diagnostics = read_jsonl(run_dir / "final_decode_diagnostics.jsonl")
    require(len(diagnostics) == len(diagnostics_expected), "final diagnostics count differs")
    for actual, expected in zip(diagnostics, diagnostics_expected):
        require(key(actual) == expected["key"], "final diagnostic key differs")
        for field in expected.keys() - {"key"}:
            require(actual[field] == expected[field], "final diagnostic differs: " + field)
    expected_counts = {"decoded_records": len(final_expected), "decoded_batches": len(grouped),
        "trigger_records": sum(r["fallback_triggered"] for r in diagnostics_expected),
        "accepted_records": sum(r["fallback_accepted"] for r in diagnostics_expected),
        "exhausted_records": sum(r["fallback_exhausted"] for r in diagnostics_expected),
        "attempt_records": len(attempt_expected), "t0_cap_hit_records": sum(r["cap_hit"] for r in t0_expected),
        "final_cap_hit_records": sum(r["cap_hit"] for r in final_expected)}
    for field, count in expected_counts.items():
        require(summary[field] == count, "summary count differs: " + field)
    return {"schema": "independent_ngram_execution_audit_v1", "pass": True,
        "run_dir": str(run_dir.resolve()), "formal_complete_coverage_required": require_complete,
        "counts": expected_counts, "generation_calls": len(calls), "member_counts": dict(member_counter),
        "checks": ["model_generation_config_only_n4_changed", "full_original_map_global_indices", "full_original_batch_members_and_positions", "seed_formula_every_sampled_call", "n4_every_generation", "prompt_inclusive_nonrepeating_token_4grams", "first_eos_and_padding", "224_new_token_budget", "token_text_roundtrip", "compression_ratios", "own_t0_trigger", "active_trajectory_and_first_acceptance", "last_attempt_adoption_on_exhaustion", "final_text_and_diagnostic_sidecars", "summary_counts"],
        "score_boundary_probe": score_boundary_probe(),
        "score_log_scope": "Saved finite sum/denominator/average values are independently checked; raw per-step logits are not retained. Independent synthetic probes verify scorer temperature normalization and EOS handling."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.run_dir, args.require_complete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
