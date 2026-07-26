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

Audio streams sentence-by-sentence as it's generated, so playback can start
before a long message has finished synthesizing -- but only on players that
actually support progressive playback of a streamed audio response (e.g.
Google Cast). Some players (notably Sonos) fully buffer the response before
playing anything, so you won't see any benefit there -- you'll still wait
for the whole message either way. Test with a Cast device if you want to
see streaming actually working.

If low latency matters more than natural intonation, "Turbo mode" further
splits just the first sentence into smaller pieces (on commas, semicolons,
colons, and dashes), so the very first sound plays noticeably sooner --
only affects the first sentence, and only while streaming.

## Settings

All of these are set at initial setup and can be changed later via the
integration's **Configure** button (options flow) -- except Model, which is
fixed once added (add a second instance if you want both Nano and Micro).

| Setting | Default | What it does |
|---|---|---|
| Model | -- | Nano (smaller/faster) or Micro (higher quality). |
| Default speed | 1.0 | Playback speed, 0.5-2.0x. Overridable per TTS call. |
| Default variation | 0.667 | How much acoustic variation between calls, 0.0-1.0. Overridable per TTS call. |
| Default seed | 7 | Random seed for that variation. Overridable per TTS call. |
| Idle unload (minutes) | 10 | Frees the model from memory after this long without a request; 0 keeps it loaded permanently. Lower this on memory-constrained hardware (e.g. a Raspberry Pi). |
| Streaming synthesis | On | Streams audio sentence-by-sentence instead of waiting for the whole message. Turn off to fall back to the old buffered behavior if a media player or HA's TTS caching misbehaves with streaming. |
| Streaming read-ahead (seconds) | 0 (auto) | How many seconds of audio a background thread may synthesize ahead of what's already been sent, while streaming -- smooths over per-sentence timing variance so sentences don't have audible gaps between them. 0 computes it automatically from your hardware's measured synthesis speed; set a fixed number to override. |
| Turbo mode | Off | Splits just the first sentence into smaller pieces for a faster time-to-first-sound while streaming, at the cost of some natural intonation on that first sentence. |

Two `inflect_tts.load_model` / `inflect_tts.unload_model` actions are also
available if you want to explicitly pre-warm or free a model from an
automation, independent of the idle-unload timer.

Two diagnostic sensors per model: **Synthesis speed** (realtime factor of
the last synthesis -- e.g. `12x` means audio generated 12x faster than its
own playback duration) and **Model load time** (how long the last request
spent loading the model into memory; `0` on a request that reused an
already-loaded model, positive right after a reload).

## Development

```
pip install -r requirements_test.txt
pip install onnxruntime phonemizer num2words Unidecode dlinfo
pytest tests/
```

## Credits

This integration is a Home Assistant wrapper around the **Inflect** text-to-speech
models by [owensong](https://huggingface.co/owensong) -- all credit for the
models themselves goes there. See the model cards for details:

- [Inflect Nano v2](https://huggingface.co/owensong/Inflect-Nano-v2)
- [Inflect Micro v2](https://huggingface.co/owensong/Inflect-Micro-v2)
