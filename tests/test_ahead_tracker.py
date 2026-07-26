"""Tests for tts.py's _AheadTracker (duration-based streaming
read-ahead) and the auto-read-ahead formula. Pure threading/math logic,
no Home Assistant runtime needed."""

from __future__ import annotations

import threading
import time

import pytest

from custom_components.inflect_tts.tts import _AheadTracker, _auto_read_ahead_seconds


def test_first_add_never_blocks() -> None:
    """The very first chunk must never be delayed -- the tally starts
    at 0, so wait_for_room() should return immediately even with a
    tiny target."""
    ahead = _AheadTracker(target_seconds=0.01)
    start = time.monotonic()
    ahead.wait_for_room()
    assert time.monotonic() - start < 0.05


def test_blocks_once_target_exceeded_and_wakes_on_remove() -> None:
    ahead = _AheadTracker(target_seconds=0.2)
    ahead.add(0.5)  # over target already

    unblocked = threading.Event()

    def waiter() -> None:
        ahead.wait_for_room()
        unblocked.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

    # Give the waiter thread a moment to actually start blocking.
    time.sleep(0.1)
    assert not unblocked.is_set()

    ahead.remove(0.5)  # back under target -- should wake it
    t.join(timeout=1.0)
    assert unblocked.is_set()


def test_remove_never_goes_negative() -> None:
    ahead = _AheadTracker(target_seconds=1.0)
    ahead.add(0.3)
    ahead.remove(10.0)  # removing more than was added
    # Should clamp at 0, not go negative and never block again.
    start = time.monotonic()
    ahead.wait_for_room()
    assert time.monotonic() - start < 0.05


@pytest.mark.parametrize(
    ("realtime_factor", "expected_range"),
    [
        (None, (2.9, 3.1)),  # no prior measurement -> conservative default
        (12.0, (0.4, 0.6)),  # comfortably fast -> minimal read-ahead
        (0.66, (4.0, 5.0)),  # slower than real-time -> more slack
        (1.34, (2.0, 2.5)),  # borderline -> moderate slack
    ],
)
def test_auto_read_ahead_seconds_scales_with_speed(
    realtime_factor: float | None, expected_range: tuple[float, float]
) -> None:
    class FakeEngine:
        last_stats = (
            {"realtime_factor": realtime_factor} if realtime_factor else None
        )

    result = _auto_read_ahead_seconds(FakeEngine())
    low, high = expected_range
    assert low <= result <= high


def test_auto_read_ahead_seconds_is_clamped_to_max() -> None:
    class FakeEngine:
        last_stats = {"realtime_factor": 0.01}  # extremely slow

    from custom_components.inflect_tts.const import MAX_STREAM_READ_AHEAD

    assert _auto_read_ahead_seconds(FakeEngine()) == MAX_STREAM_READ_AHEAD
