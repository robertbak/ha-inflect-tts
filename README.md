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

Pick a model (Nano is smaller/faster, Micro is higher quality), and
optionally set default speed/variation/seed -- these can be overridden
per-call via TTS options. The first load installs `onnxruntime` (see
above) and installs the plain-Python dependencies (`numpy`, `phonemizer`,
etc.) from PyPI as usual.

Text-to-phoneme conversion needs `espeak-ng`, a system-level dependency
that neither HACS nor pip can install -- and on real HA installs (HACS/HAOS)
there's no supported way to `apk add` something into the core container
persistently anyway, since it's rebuilt from the stock image on every
update. So a musl-built `libespeak-ng.so` (extracted from Alpine's own
package, one per architecture) ships inside this repo and is used
directly -- no system package needed at all.

## What's bundled

- `custom_components/inflect_tts/models/` -- the ONNX model graphs and
  text frontend for both models (~52MB total)
- `custom_components/inflect_tts/espeak/{x86_64,aarch64}/` -- a
  musl-built `libespeak-ng.so.1` + its data files + two small runtime
  dependencies, so phonemization works out of the box on Alpine/HAOS
  without any system package install

No separate download step at runtime for any of the above.
