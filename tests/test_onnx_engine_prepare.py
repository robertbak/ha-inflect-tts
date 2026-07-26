"""Tests for OnnxInflectEngine._prepare -- the punctuation-cue fix for
audio getting clipped at the end (see onnx_engine.py). Doesn't need
.load() or onnxruntime sessions; _prepare only touches text."""

from __future__ import annotations

import pytest

from custom_components.inflect_tts.onnx_engine import (
    InflectModelError,
    OnnxInflectEngine,
)


@pytest.fixture
def engine() -> OnnxInflectEngine:
    return OnnxInflectEngine("nano", "/nonexistent")


def test_appends_period_when_missing_terminal_punctuation(
    engine: OnnxInflectEngine,
) -> None:
    normalized, chunks = engine._prepare("turn off the lights")
    assert normalized == "turn off the lights."
    assert chunks[-1].endswith(".")


def test_does_not_double_up_existing_punctuation(
    engine: OnnxInflectEngine,
) -> None:
    normalized, _ = engine._prepare("is anyone home?")
    assert normalized == "is anyone home?"
    assert not normalized.endswith("?.")


@pytest.mark.parametrize("mark", [".", "!", "?", ";", ":"])
def test_any_terminal_punctuation_mark_is_left_alone(
    engine: OnnxInflectEngine, mark: str
) -> None:
    text = f"already ends properly{mark}"
    normalized, _ = engine._prepare(text)
    assert normalized == text


def test_collapses_internal_whitespace(engine: OnnxInflectEngine) -> None:
    normalized, _ = engine._prepare("too   many\n\tspaces")
    assert normalized == "too many spaces."


def test_empty_text_raises(engine: OnnxInflectEngine) -> None:
    with pytest.raises(InflectModelError, match="must not be empty"):
        engine._prepare("   ")
