"""Tests for onnx_engine.split_text and the punctuation-cue fix, plain
Python logic with no Home Assistant or onnxruntime dependency."""

from __future__ import annotations

from custom_components.inflect_tts.onnx_engine import (
    boundary_pause_seconds,
    split_first_sentence_turbo,
    split_text,
)


def test_splits_on_terminal_punctuation() -> None:
    text = "First sentence. Second sentence! Third one? Fourth; fifth: sixth."
    assert split_text(text) == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
        "Fourth;",
        "fifth:",
        "sixth.",
    ]


def test_does_not_split_on_commas_or_dashes_by_default() -> None:
    text = "One, two, three - four, five."
    # No terminal punctuation until the very end -- should stay a single
    # chunk; commas and dashes are not primary split points.
    assert split_text(text) == ["One, two, three - four, five."]


def test_single_sentence_no_terminal_punctuation() -> None:
    assert split_text("turn off the lights") == ["turn off the lights"]


def test_long_sentence_splits_on_comma_near_limit() -> None:
    # Force the secondary split path: one long "sentence" (no terminal
    # punctuation until the very end) that exceeds the 280-char limit,
    # with a comma conveniently placed past the halfway point.
    filler = "word " * 40  # 200 chars, no punctuation
    text = filler + "with a natural break here, " + filler + "and more text."
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 280 or " " not in c for c in chunks)
    # Rejoining should reconstruct all the original words (split_text
    # strips exactly at whitespace/punctuation boundaries).
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_long_sentence_without_punctuation_falls_back_to_space() -> None:
    text = ("word " * 100).strip() + "."  # long, no commas at all
    chunks = split_text(text)
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert len(chunk) <= 280


def test_dash_is_not_a_split_point() -> None:
    # Explicitly documents current behavior: '-' isn't a candidate in
    # the secondary split search (only ',', ';', ':' are, falling back
    # to the nearest space). Proven by swapping the dash for an
    # arbitrary letter and getting identical split points either way --
    # if '-' were special, the two would differ.
    with_dash = "a" * 150 + " - " + "b" * 150 + "."
    without_dash = "a" * 150 + " x " + "b" * 150 + "."
    assert split_text(with_dash) == [
        c.replace("x", "-") for c in split_text(without_dash)
    ]


def test_boundary_pause_seconds_by_ending_punctuation() -> None:
    assert boundary_pause_seconds("Wow!") == 0.24
    assert boundary_pause_seconds("Really?") == 0.28
    assert boundary_pause_seconds("Done.") == 0.22
    assert boundary_pause_seconds("wait;") == 0.16
    assert boundary_pause_seconds("note:") == 0.13
    assert boundary_pause_seconds("comma,") == 0.09
    assert boundary_pause_seconds("no punctuation") == 0.08


def test_turbo_splits_on_light_pause_punctuation() -> None:
    text = "This is a long sentence, with a natural pause, that could split."
    assert split_first_sentence_turbo(text) == [
        "This is a long sentence,",
        "with a natural pause,",
        "that could split.",
    ]


def test_turbo_splits_on_standalone_dash() -> None:
    text = "First part - second part - third part."
    assert split_first_sentence_turbo(text) == [
        "First part -",
        "second part -",
        "third part.",
    ]


def test_turbo_does_not_split_hyphenated_words() -> None:
    text = "A well-known state-of-the-art model with no other punctuation."
    assert split_first_sentence_turbo(text) == [text]


def test_turbo_no_split_points_returns_original_as_single_piece() -> None:
    text = "No pause points at all here"
    assert split_first_sentence_turbo(text) == [text]


def test_turbo_em_and_en_dash() -> None:
    assert split_first_sentence_turbo("First — second – third.") == [
        "First —",
        "second –",
        "third.",
    ]
