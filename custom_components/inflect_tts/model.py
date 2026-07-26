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


def get_engine(model_key: str) -> OnnxInflectEngine:
    """Return the (loading, if needed) engine for a model. Blocking."""
    with _engines_lock:
        engine = _engines.get(model_key)
        if engine is None:
            engine = OnnxInflectEngine(model_key, str(MODELS_DIR))
            engine.load()
            _engines[model_key] = engine
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
    engine = get_engine(model_key)
    data = engine.synthesize(text, speed=speed, variation=variation, seed=seed)
    return data, engine.last_stats


__all__ = [
    "InflectModelError",
    "get_engine",
    "synthesize",
    "synthesize_with_stats",
    "unload_engine",
]
