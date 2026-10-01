"""Audited average-logprob accounting for fixed-window Whisper fallback.

Hugging Face generation returns scores *after* the temperature warper.  The
fallback thresholds used by OpenAI Whisper, however, are defined on the
untempered token distribution.  This module restores that distribution before
selecting token log probabilities, validates the forced decoder prompt, keeps
the first EOS score, and applies OpenAI's ``sum / (token_count + 1)`` rule when
generation reaches its token cap without EOS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OpenAIAverageLogprob:
    """One generated sequence's audited OpenAI-style average log probability."""

    avg_logprob_openai: float
    denominator: int
    eos_logprob: float | None
    generated_step_count: int
    prompt_token_count: int
    sequence_logprob_sum: float
    temperature: float
    terminated_by_eos: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _as_vector(score: torch.Tensor, *, step: int) -> torch.Tensor:
    if not isinstance(score, torch.Tensor):
        raise TypeError(f"score step {step} is not a torch.Tensor")
    if score.ndim != 1:
        raise ValueError(
            f"score step {step} must be one vocabulary vector, got {tuple(score.shape)}"
        )
    if score.numel() == 0:
        raise ValueError(f"score step {step} has an empty vocabulary")
    return score


def openai_average_logprob_from_scaled_scores(
    sequence_ids: Sequence[int],
    scaled_scores: Sequence[torch.Tensor],
    *,
    temperature: float,
    eos_token_id: int,
    expected_prompt_ids: Sequence[int],
) -> OpenAIAverageLogprob:
    """Recover base token logprobs and compute OpenAI's fallback statistic.

    ``sequence_ids`` is one complete Hugging Face generated sequence, including
    its forced decoder prompt. ``scaled_scores`` is the corresponding per-step
    score vector sequence for that batch row (one vector per generated step).

    Hugging Face's Whisper helper multiplies sampled scores by the temperature
    before applying log-softmax. We do the same. Rows that finish early can
    contain padding after EOS; only the first EOS is retained. If no EOS was
    emitted, OpenAI's decoder would append an unscored EOS during finalization,
    so the denominator is the scored step count plus one.
    """

    value = float(temperature)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if not scaled_scores:
        raise ValueError("Whisper generation returned no score steps")

    sequence = [int(token) for token in sequence_ids]
    prompt = [int(token) for token in expected_prompt_ids]
    score_count = len(scaled_scores)
    if len(sequence) < score_count:
        raise ValueError(
            "generated sequence is shorter than its per-step score inventory"
        )
    prompt_count = len(sequence) - score_count
    if prompt_count != len(prompt):
        raise ValueError(
            "forced-prompt length and generated-score inventory disagree: "
            f"observed_prompt={prompt_count}, expected_prompt={len(prompt)}"
        )
    if sequence[:prompt_count] != prompt:
        raise ValueError(
            "Whisper forced prompt differs: "
            f"observed={sequence[:prompt_count]}, expected={prompt}"
        )

    generated = sequence[prompt_count:]
    stop = len(generated)
    terminated = False
    for index, token_id in enumerate(generated):
        if token_id == int(eos_token_id):
            stop = index + 1
            terminated = True
            break
    selected_tokens = generated[:stop]
    selected_scores = scaled_scores[:stop]
    if not selected_tokens:
        raise ValueError("Whisper generation has no scored token")

    rescale_temperature = value if value > 0.0 else 1.0
    token_logprobs: list[float] = []
    for step, (token_id, score) in enumerate(
        zip(selected_tokens, selected_scores, strict=True)
    ):
        vector = _as_vector(score, step=step)
        if token_id < 0 or token_id >= vector.numel():
            raise ValueError(
                f"selected token {token_id} is outside score vocabulary "
                f"at step {step}"
            )
        base_logprobs = F.log_softmax(
            (vector * rescale_temperature).float(), dim=-1
        )
        selected = float(base_logprobs[token_id].item())
        if not math.isfinite(selected):
            raise ValueError(
                f"selected token has non-finite base logprob at step {step}"
            )
        token_logprobs.append(selected)

    sequence_sum = float(sum(token_logprobs))
    denominator = len(token_logprobs) if terminated else len(token_logprobs) + 1
    return OpenAIAverageLogprob(
        avg_logprob_openai=sequence_sum / denominator,
        denominator=denominator,
        eos_logprob=(token_logprobs[-1] if terminated else None),
        generated_step_count=len(token_logprobs),
        prompt_token_count=prompt_count,
        sequence_logprob_sum=sequence_sum,
        temperature=value,
        terminated_by_eos=terminated,
    )


def openai_average_logprob_batch_from_scaled_scores(
    sequence_ids: Sequence[Sequence[int]] | torch.Tensor,
    scaled_scores: Sequence[torch.Tensor],
    *,
    temperature: float,
    eos_token_id: int,
    expected_prompt_ids: Sequence[int],
) -> list[OpenAIAverageLogprob]:
    """Vectorized batch equivalent of the single-row audited oracle.

    Each score step is normalized once as a ``[batch, vocabulary]`` matrix.
    This avoids repeating a full-vocabulary log-softmax for every batch row.
    """

    value = float(temperature)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    if not scaled_scores:
        raise ValueError("Whisper generation returned no score steps")
    sequences = torch.as_tensor(sequence_ids, dtype=torch.long)
    if sequences.ndim != 2 or sequences.shape[0] == 0:
        raise ValueError("generated sequences must be a non-empty rank-2 batch")
    score_count = len(scaled_scores)
    if sequences.shape[1] < score_count:
        raise ValueError(
            "generated sequence is shorter than its per-step score inventory"
        )
    prompt = [int(token) for token in expected_prompt_ids]
    prompt_count = int(sequences.shape[1]) - score_count
    if prompt_count != len(prompt):
        raise ValueError(
            "forced-prompt length and generated-score inventory disagree: "
            f"observed_prompt={prompt_count}, expected_prompt={len(prompt)}"
        )
    expected = torch.tensor(prompt, dtype=torch.long)
    if not torch.equal(
        sequences[:, :prompt_count], expected.expand(sequences.shape[0], -1)
    ):
        raise ValueError("Whisper forced prompt differs in batch")
    generated = sequences[:, prompt_count:]
    batch = int(sequences.shape[0])
    rescale_temperature = value if value > 0.0 else 1.0
    selected_steps: list[torch.Tensor] = []
    for step, score in enumerate(scaled_scores):
        if not isinstance(score, torch.Tensor) or score.ndim != 2:
            raise ValueError(f"score step {step} must be [batch, vocabulary]")
        if int(score.shape[0]) != batch or int(score.shape[1]) <= 0:
            raise ValueError(f"score step {step} shape differs from sequence batch")
        token = generated[:, step].to(score.device)
        if bool(torch.any(token < 0)) or bool(torch.any(token >= score.shape[1])):
            raise ValueError(f"selected token is outside vocabulary at step {step}")
        base = score.float() * rescale_temperature
        chosen = base.gather(1, token[:, None]).squeeze(1)
        selected = chosen - torch.logsumexp(base, dim=-1)
        selected_steps.append(selected.detach().cpu())
    selected_matrix = torch.stack(selected_steps, dim=1)
    if not bool(torch.isfinite(selected_matrix).all()):
        raise ValueError("selected token has non-finite base logprob")

    output: list[OpenAIAverageLogprob] = []
    generated_cpu = generated.cpu()
    for row in range(batch):
        tokens = generated_cpu[row].tolist()
        stop = score_count
        terminated = False
        for index, token_id in enumerate(tokens):
            if int(token_id) == int(eos_token_id):
                stop = index + 1
                terminated = True
                break
        logprobs = selected_matrix[row, :stop]
        sequence_sum = float(logprobs.sum().item())
        denominator = stop if terminated else stop + 1
        output.append(
            OpenAIAverageLogprob(
                avg_logprob_openai=sequence_sum / denominator,
                denominator=denominator,
                eos_logprob=(float(logprobs[-1].item()) if terminated else None),
                generated_step_count=stop,
                prompt_token_count=prompt_count,
                sequence_logprob_sum=sequence_sum,
                temperature=value,
                terminated_by_eos=terminated,
            )
        )
    return output