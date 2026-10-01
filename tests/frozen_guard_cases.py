from realmeetsep.asr_repetition_guard_v035 import guard_repetitions


def test_ordinary_short_repetition_is_unchanged() -> None:
    text = "对对对我觉得这个方案很好"
    result = guard_repetitions(text)
    assert result.text == text
    assert result.events == ()


def test_long_two_character_loop_is_reduced() -> None:
    result = guard_repetitions("正常开头" + "我们" * 100)
    assert result.text == "正常开头" + "我们" * 2
    assert result.removed_characters == 196
    assert result.events[0].period_chars == 2


def test_internal_loop_preserves_suffix() -> None:
    result = guard_repetitions("开头" + "我也是" * 12 + "结尾")
    assert result.text == "开头" + "我也是" * 2 + "结尾"


def test_multiple_loops_and_idempotence() -> None:
    text = "甲" * 30 + "中间" + "乙丙" * 20
    first = guard_repetitions(text)
    second = guard_repetitions(first.text)
    assert first.text == "甲" * 2 + "中间" + "乙丙" * 2
    assert len(first.events) == 2
    assert second.text == first.text
    assert second.events == ()


def test_parameter_validation() -> None:
    try:
        guard_repetitions("x", keep_repetitions=6, min_repetitions=6)
    except ValueError as error:
        assert "keep_repetitions" in str(error)
    else:
        raise AssertionError("invalid parameters were accepted")

