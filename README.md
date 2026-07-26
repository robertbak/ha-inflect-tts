# Inflect TTS for Home Assistant

A local [text-to-speech](https://www.home-assistant.io/integrations/tts/)
integration for Home Assistant. Runs fully on-device -- no cloud calls, no
internet dependency, no API keys.

## What you get

- Two voices to choose from: **Nano** (smaller and faster) and **Micro**
  (higher quality)
- Adjustable speed, variation, and seed, either as defaults or per TTS call
- A diagnostic sensor showing how fast synthesis is running on your hardware
- `load_model` / `unload_model` actions to pre-warm or free the model on
  demand, and an automatic idle-unload timer to keep memory use low on
  low-end hardware like a Raspberry Pi
- Works on both x86_64 and ARM installs

## Installation

### HACS (custom repository)

1. HACS -> the three-dot menu (top right) -> **Custom repositories**
2. Repository: `https://github.com/robertbak/ha-inflect-tts`, Type: **Integration**
3. Install "Inflect TTS", restart Home Assistant
4. Settings -> Devices & Services -> Add Integration -> "Inflect TTS"

### Manual

Copy `custom_components/inflect_tts` into your Home Assistant
`config/custom_components/` directory, restart, then add the integration
as above.

## Setup

Pick a model (Nano or Micro) and, optionally, default speed/variation/seed.
Everything the integration needs is installed automatically on first load --
no extra setup steps.

## Credits

This integration is a Home Assistant wrapper around the **Inflect** text-to-speech
models by [owensong](https://huggingface.co/owensong) -- all credit for the
models themselves goes there. See the model cards for details:

- [Inflect Nano v2](https://huggingface.co/owensong/Inflect-Nano-v2)
- [Inflect Micro v2](https://huggingface.co/owensong/Inflect-Micro-v2)
