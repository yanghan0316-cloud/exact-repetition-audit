from __future__ import annotations

from dataclasses import dataclass

from realmeetsep.asr_delooping_baselines_v046_postprimary import (
    collapse_ltr_character_repetitions,
    collapse_whisper_token_repetitions,
)


@dataclass
class CharacterTokenizer:
    """Deterministic one-code-point tokenizer for unit tests."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }


class OverlappingOffsetTokenizer:
    """Represent one Unicode character with overlapping byte-piece offsets."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, object]:
        return {
            "input_ids": [1, 2, 3] * len(text),
            "offset_mapping": [
                (character_index, character_index + 1)
                for character_index in range(len(text))
                for _ in range(3)
            ],
        }


def test_ltr_character_collapses_exact_run_and_is_idempotent() -> None:
    text = "开头" + "甲乙丙丁" * 12 + "结尾"
    first = collapse_ltr_character_repetitions(text)
    second = collapse_ltr_character_repetitions(first.text)
    assert first.text == "开头" + "甲乙丙丁" * 2 + "结尾"
    assert first.removed_characters == 40
    assert len(first.events) == 1
    assert first.events[0].period_units == 4
    assert second.text == first.text
    assert second.events == ()


def test_ltr_policy_chooses_earliest_run_before_larger_later_run() -> None:
    text = "abcd" * 6 + "中间" + "uvwxyz" * 20
    result = collapse_ltr_character_repetitions(text)
    assert len(result.events) == 2
    assert result.events[0].original_start_char == 0
    assert result.events[0].period_units == 4
    assert result.events[1].period_units == 6


def test_ltr_no_event_preserves_exact_input_object_value() -> None:
    text = " ordinary 文字，含 punctuation\tand spacing "
    result = collapse_ltr_character_repetitions(text)
    assert result.text == text
    assert result.events == ()
    assert result.removed_characters == 0


def test_token_baseline_with_character_tokenizer() -> None:
    tokenizer = CharacterTokenizer()
    text = "前" + "甲乙丙丁" * 10 + "后"
    first = collapse_whisper_token_repetitions(text, tokenizer=tokenizer)
    second = collapse_whisper_token_repetitions(
        first.text, tokenizer=tokenizer
    )
    assert first.text == "前" + "甲乙丙丁" * 2 + "后"
    assert first.removed_characters == 32
    assert len(first.events) == 1
    assert first.events[0].unit_kind == "whisper_content_token"
    assert first.events[0].period_units == 4
    assert second.text == first.text
    assert second.events == ()


def test_token_no_event_is_not_round_tripped_through_decode() -> None:
    tokenizer = CharacterTokenizer()
    text = "  原样保留\tA B  "
    result = collapse_whisper_token_repetitions(text, tokenizer=tokenizer)
    assert result.text == text
    assert result.events == ()


def test_token_baseline_skips_unsafe_byte_piece_period() -> None:
    result = collapse_whisper_token_repetitions(
        "界" * 12,
        tokenizer=OverlappingOffsetTokenizer(),
    )
    # One- and two-token candidates would cut a Unicode character.  The first
    # admissible period is the complete three-piece character.
    assert result.text == "界" * 2
    assert len(result.events) == 1
    assert result.events[0].period_units == 3


def test_parameter_validation() -> None:
    try:
        collapse_ltr_character_repetitions(
            "x", min_repetitions=6, keep_repetitions=6
        )
    except ValueError as error:
        assert "keep_repetitions" in str(error)
    else:
        raise AssertionError("invalid parameters were accepted")

