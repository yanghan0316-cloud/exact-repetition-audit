"""Checks for the independent post-primary token n-gram blocking experiment.

The average log probability is computed from processed generation scores,
undoing temperature while retaining the n-gram/suppression constraints.  Its
meaning is therefore the constrained distribution, not raw model confidence.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from realmeetsep.asr_whisper_fallback_v039 import OpenAIAverageLogprob


def constrained_average_logprob_batch(
    sequence_ids: Sequence[Sequence[int]] | torch.Tensor,
    scaled_scores: Sequence[torch.Tensor],
    *,
    temperature: float,
    eos_token_id: int,
    expected_prompt_ids: Sequence[int],
) -> list[OpenAIAverageLogprob]:
    """V039 arithmetic with finite checks restricted through the first EOS.

EOS and padding share an ID in Whisper. Scores of padding after the first EOS
can legitimately be -inf under n-gram blocking; they are never observations.
The actual first EOS remains scored and must have a finite log probability.
"""
    value = float(temperature)
    if not math.isfinite(value) or value < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not scaled_scores:
        raise ValueError("Whisper generation returned no score steps")
    sequences = torch.as_tensor(sequence_ids, dtype=torch.long).detach().cpu()
    if sequences.ndim != 2 or sequences.shape[0] == 0:
        raise ValueError("generated sequences must be a non-empty rank-2 batch")
    prompt = [int(token) for token in expected_prompt_ids]
    score_count = len(scaled_scores)
    if int(sequences.shape[1]) - score_count != len(prompt):
        raise ValueError("forced-prompt length and generated-score inventory disagree")
    expected = torch.tensor(prompt, dtype=torch.long)
    if not torch.equal(sequences[:, :len(prompt)], expected.expand(sequences.shape[0], -1)):
        raise ValueError("Whisper forced prompt differs in batch")
    generated = sequences[:, len(prompt):]
    batch = int(sequences.shape[0])
    stops: list[int] = []
    terminated: list[bool] = []
    for tokens in generated.tolist():
        eos_positions = [i for i, token in enumerate(tokens) if token == eos_token_id]
        terminated.append(bool(eos_positions))
        stops.append(eos_positions[0] + 1 if eos_positions else score_count)
    scale = value if value > 0 else 1.0
    selected_steps: list[torch.Tensor] = []
    for step, score in enumerate(scaled_scores):
        if not isinstance(score, torch.Tensor) or score.ndim != 2:
            raise ValueError(f"score step {step} must be [batch, vocabulary]")
        if int(score.shape[0]) != batch or int(score.shape[1]) <= 0:
            raise ValueError(f"score step {step} shape differs from sequence batch")
        token = generated[:, step].to(score.device)
        if bool(torch.any(token < 0)) or bool(torch.any(token >= score.shape[1])):
            raise ValueError(f"selected token is outside vocabulary at step {step}")
        base = score.float() * scale
        selected = base.gather(1, token[:, None]).squeeze(1) - torch.logsumexp(base, dim=-1)
        selected_steps.append(selected.detach().cpu())
    matrix = torch.stack(selected_steps, dim=1)
    output: list[OpenAIAverageLogprob] = []
    for row, (stop, eos) in enumerate(zip(stops, terminated, strict=True)):
        logprobs = matrix[row, :stop]
        if not bool(torch.isfinite(logprobs).all()):
            raise ValueError(f"selected token has non-finite constrained logprob in valid row {row}")
        total = float(logprobs.sum().item())
        denominator = stop if eos else stop + 1
        output.append(OpenAIAverageLogprob(
            avg_logprob_openai=total / denominator,
            denominator=denominator,
            eos_logprob=float(logprobs[-1].item()) if eos else None,
            generated_step_count=stop,
            prompt_token_count=len(prompt),
            sequence_logprob_sum=total,
            temperature=value,
            terminated_by_eos=eos,
        ))
    return output


def validate_token_history(
    sequence_ids: Sequence[int],
    *,
    prompt_ids: Sequence[int],
    score_count: int,
    eos_token_id: int,
    pad_token_id: int,
    ngram_size: int,
    max_new_tokens: int,
    timestamp_begin: int,
) -> dict[str, object]:
    """Check the same prompt-inclusive history used by HF's processor."""
    sequence = [int(value) for value in sequence_ids]
    prompt = [int(value) for value in prompt_ids]
    if len(prompt) != 4 or sequence[:len(prompt)] != prompt:
        raise ValueError("expected the fixed four-token Mandarin transcription prompt")
    if len(sequence) - len(prompt) != score_count:
        raise ValueError("sequence and score-step counts disagree")
    if not 1 <= score_count <= max_new_tokens:
        raise ValueError("generation exceeds its token budget or has no scored tokens")
    if ngram_size < 1:
        raise ValueError("ngram_size must be positive")
    generated = sequence[len(prompt):]
    terminated = eos_token_id in generated
    stop = generated.index(eos_token_id) + 1 if terminated else len(generated)
    if any(token != pad_token_id for token in generated[stop:]):
        raise ValueError("non-padding token follows the first EOS")
    valid = generated[:stop]
    if not terminated and len(valid) != max_new_tokens:
        raise ValueError("generation stopped without EOS before reaching the budget")
    history = list(prompt)
    seen = {tuple(history[i:i + ngram_size]) for i in range(len(history) - ngram_size + 1)}
    for token in valid:
        history.append(token)
        if len(history) >= ngram_size:
            gram = tuple(history[-ngram_size:])
            if gram in seen:
                raise ValueError(f"repeated {ngram_size}-gram in valid prompt-inclusive token history: {gram}")
            seen.add(gram)
    return {
        "sequence_token_ids": sequence,
        "valid_generated_token_ids": valid,
        "generated_step_count": len(valid),
        "score_step_count": score_count,
        "terminated_by_eos": terminated,
        "cap_hit": len(valid) == max_new_tokens,
        "no_eos_at_cap": not terminated,
        "ngram_constraint_verified": True,
        "ngram_history_includes_prompt": True,
        "generated_timestamp_token": any(token >= timestamp_begin for token in valid),
    }
