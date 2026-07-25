# Inflect TTS for Home Assistant

A local, in-process [Text-to-speech](https://www.home-assistant.io/integrations/tts/)
integration for Home Assistant using the
[Inflect Micro/Nano v2](https://huggingface.co/owensong/Inflect-Nano-v2)
models, exported to ONNX. Runs directly inside Home Assistant -- no sidecar
container, no torch, no cloud calls.

## Why ONNX instead of torch

Home Assistant's official container images are Alpine Linux (musl libc).
`torch` has no musllinux wheel and is impractical to build on musl (hard
dependency on MKL, which is glibc-only). `onnxruntime` has no official
musllinux wheel either, but unlike torch it's practical to build from
source -- see
[robertbak/onnxruntime-musllinux-wheels](https://github.com/robertbak/onnxruntime-musllinux-wheels)
for prebuilt wheels (x86_64 + aarch64) and the build scripts. This
integration installs the right one automatically the first time it loads.

## Installation

### HACS (custom repository)

1. HACS -> the three-dot menu (top right) -> **Custom repositories**
2. Repository: `https://github.com/robertbak/ha-inflect-tts`, Type: **Integration**
3. Install "Inflect TTS", restart Home Assistant
4. Settings -> Devices & Services -> Add Integration -> "Inflect TTS"

### Manual

Copy `custom_components/inflect_tts` into your Home Assistant `config/custom_components/`
directory, restart, then add the integration as above.

## Setup

Pick a model (Micro is smaller/faster, Nano is higher quality), and
optionally set default speed/variation/seed -- these can be overridden
per-call via TTS options. The first load installs `onnxruntime` (see
above) and installs the plain-Python dependencies (`numpy`, `phonemizer`,
etc.) from PyPI as usual.

Requires `espeak-ng` on the host/container for text-to-phoneme conversion
(`apk add espeak-ng` on Alpine, `apt-get install espeak-ng` on Debian-based
images) -- not something HACS/pip can install, since it's a system package.

## What's bundled

The ONNX model graphs and text frontend for both models ship inside this
repo under `custom_components/inflect_tts/models/` (~52MB total) -- no
separate download step at runtime.
