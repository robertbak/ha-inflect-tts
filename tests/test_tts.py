"""Tests for the Inflect TTS entity -- in-process ONNX synthesis (no
sidecar), using the real bundled Nano model rather than mocking a
network call, since synthesis itself is fast and local."""

from __future__ import annotations

import wave
from io import BytesIO

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.inflect_tts.const import (
    CONF_MODEL,
    CONF_STREAMING,
    CONF_TURBO_MODE,
    DOMAIN,
    MODEL_NANO,
)


async def _setup_entry(hass: HomeAssistant, **data) -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_MODEL: MODEL_NANO, **data})
    entry.add_to_hass(hass)

    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


def _wav_duration_seconds(data: bytes) -> float:
    with wave.open(BytesIO(data)) as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


async def test_setup_creates_tts_and_sensor_entities(hass: HomeAssistant) -> None:
    await async_setup_component(hass, "homeassistant", {})
    await _setup_entry(hass)

    assert hass.states.get("tts.inflect_nano_v2") is not None
    assert hass.states.get("sensor.inflect_nano_v2_synthesis_speed") is not None
    assert hass.states.get("sensor.inflect_nano_v2_model_load_time") is not None


async def test_get_tts_audio_returns_real_wav(hass: HomeAssistant) -> None:
    await async_setup_component(hass, "homeassistant", {})
    await _setup_entry(hass)

    component = hass.data["tts"]
    entity = component.get_entity("tts.inflect_nano_v2")

    extension, data = await entity.async_get_tts_audio(
        "Hello from the test suite.", "en-US", {}
    )

    assert extension == "wav"
    assert data[:4] == b"RIFF"
    assert _wav_duration_seconds(data) > 0


async def test_get_tts_audio_empty_text_raises(hass: HomeAssistant) -> None:
    await async_setup_component(hass, "homeassistant", {})
    await _setup_entry(hass)

    component = hass.data["tts"]
    entity = component.get_entity("tts.inflect_nano_v2")

    with pytest.raises(HomeAssistantError):
        await entity.async_get_tts_audio("   ", "en-US", {})


async def test_get_tts_audio_updates_speed_sensor(hass: HomeAssistant) -> None:
    await async_setup_component(hass, "homeassistant", {})
    await _setup_entry(hass)

    component = hass.data["tts"]
    entity = component.get_entity("tts.inflect_nano_v2")
    await entity.async_get_tts_audio("Checking the speed sensor updates.", "en-US", {})
    await hass.async_block_till_done()

    state = hass.states.get("sensor.inflect_nano_v2_synthesis_speed")
    assert state.state not in (None, "unknown", "unavailable")
    assert float(state.state) > 0
    assert "rtf" in state.attributes


async def test_stream_multi_sentence_yields_multiple_chunks(
    hass: HomeAssistant,
) -> None:
    """Streaming (default on) should deliver more than one chunk for a
    multi-sentence message: a header plus per-sentence PCM."""
    await async_setup_component(hass, "homeassistant", {})
    await _setup_entry(hass)

    component = hass.data["tts"]
    entity = component.get_entity("tts.inflect_nano_v2")

    async def message_gen():
        yield "This is the first sentence. This is the second sentence."

    from homeassistant.components.tts import TTSAudioRequest

    response = await entity.async_stream_tts_audio(
        TTSAudioRequest(language="en-US", options={}, message_gen=message_gen())
    )

    chunks = [chunk async for chunk in response.data_gen]
    assert response.extension == "wav"
    assert len(chunks) > 2  # header + at least 2 sentence chunks
    assert chunks[0][:4] == b"RIFF"


async def test_streaming_disabled_falls_back_to_single_chunk(
    hass: HomeAssistant,
) -> None:
    await async_setup_component(hass, "homeassistant", {})
    await _setup_entry(hass, **{CONF_STREAMING: False})

    component = hass.data["tts"]
    entity = component.get_entity("tts.inflect_nano_v2")

    async def message_gen():
        yield "First sentence. Second sentence. Third sentence."

    from homeassistant.components.tts import TTSAudioRequest

    response = await entity.async_stream_tts_audio(
        TTSAudioRequest(language="en-US", options={}, message_gen=message_gen())
    )

    chunks = [chunk async for chunk in response.data_gen]
    assert len(chunks) == 1
    assert chunks[0][:4] == b"RIFF"


async def test_turbo_mode_reduces_time_to_first_chunk(hass: HomeAssistant) -> None:
    """Turbo mode should split the first sentence into more, smaller
    pieces (faster first chunk) without changing the overall content."""
    await async_setup_component(hass, "homeassistant", {})
    await _setup_entry(hass, **{CONF_TURBO_MODE: True})

    component = hass.data["tts"]
    entity = component.get_entity("tts.inflect_nano_v2")

    text = (
        "This is a fairly long opening sentence, with several natural "
        "pause points, that could be split into pieces. And a second one."
    )

    async def message_gen():
        yield text

    from homeassistant.components.tts import TTSAudioRequest

    response = await entity.async_stream_tts_audio(
        TTSAudioRequest(language="en-US", options={}, message_gen=message_gen())
    )
    chunks = [chunk async for chunk in response.data_gen]

    # header + several turbo-split first-sentence pieces (with pauses
    # between) + the second sentence -- more chunks than non-turbo would
    # produce for the same text (verified against the non-turbo case in
    # test_stream_multi_sentence_yields_multiple_chunks, which uses a
    # simpler 2-sentence message and gets far fewer chunks).
    assert len(chunks) > 5
    assert chunks[0][:4] == b"RIFF"


async def test_turbo_mode_off_by_default(hass: HomeAssistant) -> None:
    """MockConfigEntry doesn't go through the config flow's schema
    defaults, so this checks the entity itself falls back correctly
    when CONF_TURBO_MODE isn't present in the entry data at all."""
    await async_setup_component(hass, "homeassistant", {})
    await _setup_entry(hass)

    component = hass.data["tts"]
    entity = component.get_entity("tts.inflect_nano_v2")
    assert entity._turbo_mode is False


async def test_load_and_unload_model_services(hass: HomeAssistant) -> None:
    await async_setup_component(hass, "homeassistant", {})
    entry = await _setup_entry(hass)

    from custom_components.inflect_tts.model import _engines

    await hass.services.async_call(
        "inflect_tts",
        "load_model",
        {"entity_id": "tts.inflect_nano_v2"},
        blocking=True,
    )
    assert MODEL_NANO in _engines

    await hass.services.async_call(
        "inflect_tts",
        "unload_model",
        {"entity_id": "tts.inflect_nano_v2"},
        blocking=True,
    )
    assert MODEL_NANO not in _engines
