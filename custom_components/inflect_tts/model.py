"""In-process ONNX inference for the Inflect TTS models.

Used to require a separate sidecar container because onnxruntime had no
musllinux (Alpine) wheel and HA's official container is Alpine-based.
Building one from source (see ../../sidecar/musl-wheel-build/) removed
that blocker, so this runs directly inside Home Assistant now.

Model artifacts (ONNX graphs + text frontend) live in ./models/<key>/,
copied from what the sidecar's own export stage produces -- see
../../sidecar/export/export_onnx.py.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator

from .const import MODELS_DIR
from .onnx_engine import InflectModelError, OnnxInflectEngine

_engines: dict[str, OnnxInflectEngine] = {}
# Guards check-then-load in get_engine and pop in unload_engine -- without
# it, two near-simultaneous first requests (or a request racing the
# idle-unload timer) could double-load or unload out from under an
# in-flight load. Only held around the dict/load bookkeeping, never
# around engine.synthesize() itself, so concurrent synthesis calls on an
# already-loaded engine still run unserialized.
_engines_lock = threading.Lock()


def _get_or_load(model_key: str) -> tuple[OnnxInflectEngine, bool]:
    """Return (engine, loaded_fresh) -- loaded_fresh tells the caller
    whether this call just paid the cold-start cost (e.g. after the
    idle-unload timer freed it), so callers that report stats can
    surface that separately from steady-state synthesis time."""
    with _engines_lock:
        engine = _engines.get(model_key)
        if engine is None:
            engine = OnnxInflectEngine(model_key, str(MODELS_DIR))
            engine.load()
            _engines[model_key] = engine
            return engine, True
        return engine, False


def get_engine(model_key: str) -> OnnxInflectEngine:
    """Return the (loading, if needed) engine for a model. Blocking."""
    engine, _ = _get_or_load(model_key)
    return engine


def unload_engine(model_key: str) -> None:
    """Drop a cached engine so its ONNX sessions can be garbage collected.
    Call when a config entry using it is unloaded/removed -- otherwise
    reconfiguring keeps every past session alive in memory.
    """
    with _engines_lock:
        _engines.pop(model_key, None)


def synthesize(
    model_key: str,
    text: str,
    speed: float,
    variation: float,
    seed: int,
) -> bytes:
    """Run a full synthesis pass. Blocking -- call via
    hass.async_add_executor_job, never directly from the event loop.
    """
    engine = get_engine(model_key)
    return engine.synthesize(text, speed=speed, variation=variation, seed=seed)


def synthesize_with_stats(
    model_key: str,
    text: str,
    speed: float,
    variation: float,
    seed: int,
) -> tuple[bytes, dict]:
    """Same as synthesize(), but also returns the last-synthesis timing
    stats from the same engine call -- atomic with the synthesis itself,
    so there's no race with the idle-unload timer between calls.
    """
    engine, loaded_fresh = _get_or_load(model_key)
    data = engine.synthesize(text, speed=speed, variation=variation, seed=seed)
    stats = dict(engine.last_stats) if engine.last_stats is not None else None
    if stats is not None:
        stats["cold_start_seconds"] = (
            engine.last_load_seconds if loaded_fresh else 0.0
        )
    return data, stats


def get_stream(
    model_key: str,
    text: str,
    speed: float,
    variation: float,
    seed: int,
    turbo: bool = False,
) -> tuple[OnnxInflectEngine, Iterator[bytes], bool]:
    """Load the engine (if needed) and return it along with a ready-to
    -iterate generator of raw PCM16 chunks, and whether this call just
    paid the cold-start load cost. Blocking -- call via
    hass.async_add_executor_job. The generator itself is lazy (creating
    it doesn't run any inference), so only this initial call needs the
    executor for engine loading; each subsequent chunk still needs its
    own executor call to advance the generator, since each one blocks
    on model inference.

    The returned engine is also the caller's cue for stats: read
    engine.last_stats once the generator is exhausted.
    """
    engine, loaded_fresh = _get_or_load(model_key)
    return (
        engine,
        engine.synthesize_stream(
            text, speed=speed, variation=variation, seed=seed, turbo=turbo
        ),
        loaded_fresh,
    )


__all__ = [
    "InflectModelError",
    "get_engine",
    "get_stream",
    "synthesize",
    "synthesize_with_stats",
    "unload_engine",
]
