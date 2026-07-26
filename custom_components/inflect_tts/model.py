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

from .const import MODELS_DIR
from .onnx_engine import InflectModelError, OnnxInflectEngine

_engines: dict[str, OnnxInflectEngine] = {}


def get_engine(model_key: str) -> OnnxInflectEngine:
    """Return the (loading, if needed) engine for a model. Blocking."""
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


__all__ = ["InflectModelError", "get_engine", "synthesize", "unload_engine"]
