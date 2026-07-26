"""Constants for the Inflect TTS integration."""

from pathlib import Path

DOMAIN = "inflect_tts"

# Deliberately defined here rather than in model.py: config_flow.py needs
# this path (to check a model's artifacts exist) without importing
# onnx_engine.py, which imports onnxruntime at module level -- onnxruntime
# may not be installed yet the first time config_flow loads (see
# onnxruntime_install.py), and config_flow is imported before any config
# entry (and thus before __init__.py's async_setup_entry) ever runs.
MODELS_DIR = Path(__file__).parent / "models"

CONF_MODEL = "model"
CONF_SPEED = "speed"
CONF_VARIATION = "variation"
CONF_SEED = "seed"
CONF_IDLE_UNLOAD_MINUTES = "idle_unload_minutes"
CONF_STREAMING = "streaming"

MODEL_MICRO = "micro"
MODEL_NANO = "nano"

MODEL_REPOS = {
    MODEL_MICRO: "owensong/Inflect-Micro-v2",
    MODEL_NANO: "owensong/Inflect-Nano-v2",
}

MODEL_NAMES = {
    MODEL_MICRO: "Inflect Micro v2",
    MODEL_NANO: "Inflect Nano v2",
}

DEFAULT_MODEL = MODEL_NANO
DEFAULT_SPEED = 1.0
DEFAULT_VARIATION = 0.667
DEFAULT_SEED = 7
# Unload the ONNX sessions after this many idle minutes to free memory on
# low-end hardware (e.g. Raspberry Pi); 0 disables idle unloading and keeps
# the model resident once loaded, trading memory for lower latency.
DEFAULT_IDLE_UNLOAD_MINUTES = 10
# HA always calls the streaming entry point -- there's no per-call choice
# for the end user -- so this is an escape hatch to fall back to the old
# buffered (whole-message) behavior if streaming misbehaves with a given
# media player or with HA's own TTS caching.
DEFAULT_STREAMING = True

MIN_SPEED = 0.5
MAX_SPEED = 2.0
MIN_VARIATION = 0.0
MAX_VARIATION = 1.0
MIN_IDLE_UNLOAD_MINUTES = 0
MAX_IDLE_UNLOAD_MINUTES = 240

SUPPORT_LANGUAGES = ["en-US"]
DEFAULT_LANG = "en-US"

SAMPLE_RATE = 24000

# Upstream onnxruntime has no musllinux (Alpine) wheel, which is what HA's
# own container runs on -- these are built from source (see
# ../../sidecar/musl-wheel-build/) and served as a PEP 503 index covering
# both architectures HA runs on, so pip resolves the right one on its own.
ONNXRUNTIME_VERSION = "1.28.0"
ONNXRUNTIME_INDEX_URL = "https://robertbak.github.io/onnxruntime-musllinux-wheels/"
