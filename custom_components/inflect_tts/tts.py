"""Support for the Inflect local TTS models (Micro/Nano), in-process."""

from __future__ import annotations

import logging
import queue as sync_queue
import struct
import threading
from collections.abc import Iterator
from typing import Any

from homeassistant.components.tts import (
    TextToSpeechEntity,
    TTSAudioRequest,
    TTSAudioResponse,
    TtsAudioType,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers import entity_platform
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later

from .const import (
    CONF_IDLE_UNLOAD_MINUTES,
    CONF_MODEL,
    CONF_SEED,
    CONF_SPEED,
    CONF_STREAM_READ_AHEAD,
    CONF_STREAMING,
    CONF_TURBO_MODE,
    CONF_VARIATION,
    DEFAULT_IDLE_UNLOAD_MINUTES,
    DEFAULT_LANG,
    DEFAULT_SEED,
    DEFAULT_SPEED,
    DEFAULT_STREAM_READ_AHEAD,
    DEFAULT_STREAMING,
    DEFAULT_TURBO_MODE,
    DEFAULT_VARIATION,
    DOMAIN,
    MAX_STREAM_READ_AHEAD,
    MODEL_NAMES,
    SUPPORT_LANGUAGES,
    _AUTO_READ_AHEAD_TARGET_SECONDS,
)
from .model import (
    InflectModelError,
    OnnxInflectEngine,
    get_engine,
    get_stream,
    synthesize_with_stats,
    unload_engine,
)

SERVICE_LOAD_MODEL = "load_model"
SERVICE_UNLOAD_MODEL = "unload_model"

_STREAM_DONE = object()


def _streaming_wav_header(
    sample_rate: int, channels: int = 1, bits_per_sample: int = 16
) -> bytes:
    """A canonical 44-byte PCM WAV header with the size fields set to
    the placeholder "unknown length" value, since the total duration
    isn't known until the whole stream (all sentence chunks) has been
    generated. Players/consumers that support streamed WAV treat this
    as "play until the stream ends" rather than validating the exact
    byte count -- the same trick used for e.g. streamed internet radio.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    unknown_size = 0xFFFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        unknown_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        unknown_size,
    )


_BYTES_PER_SAMPLE = 2  # 16-bit PCM


class _AheadTracker:
    """Paces the producer thread by seconds of buffered *audio*, not
    chunk count -- a 10s sentence and ten 1s sentences shouldn't count
    the same. Every PCM chunk's exact duration is free to compute
    (len(chunk) / bytes_per_sample / sample_rate), so this is just
    tracking that running total: the producer blocks once it's built up
    target_seconds of unconsumed audio, and wakes up again as the
    consumer hands chunks off (removes them from the tally). The first
    chunk is never delayed -- the tally starts at 0.
    """

    def __init__(self, target_seconds: float) -> None:
        self._target_seconds = target_seconds
        self._buffered_seconds = 0.0
        self._condition = threading.Condition()

    def wait_for_room(self) -> None:
        with self._condition:
            while self._buffered_seconds >= self._target_seconds:
                self._condition.wait()

    def add(self, seconds: float) -> None:
        with self._condition:
            self._buffered_seconds += seconds
            self._condition.notify_all()

    def remove(self, seconds: float) -> None:
        with self._condition:
            self._buffered_seconds = max(0.0, self._buffered_seconds - seconds)
            self._condition.notify_all()


def _drain_chunks_into_queue(
    chunk_gen: Iterator[bytes],
    out_queue: sync_queue.Queue,
    ahead: _AheadTracker,
    sample_rate: int,
) -> None:
    """Runs in a background executor thread for the whole stream's
    duration (not per-chunk), continuously pulling chunks from the
    blocking generator and pushing them into an (unbounded -- pacing is
    ahead's job, not the queue's) queue. This is what actually gives us
    read-ahead: since this thread keeps running while the consumer is
    busy handing a chunk off to HA/the network, synthesis of the NEXT
    sentence overlaps with transmission of the current one instead of
    only starting once the consumer asks for it.
    """
    try:
        for chunk in chunk_gen:
            ahead.wait_for_room()
            duration = len(chunk) / _BYTES_PER_SAMPLE / sample_rate
            out_queue.put((chunk, duration))
            ahead.add(duration)
    except InflectModelError as exc:
        out_queue.put(exc)
    except Exception as exc:  # noqa: BLE001
        out_queue.put(InflectModelError(f"Synthesis failed: {exc}"))
    finally:
        out_queue.put(_STREAM_DONE)


def _auto_read_ahead_seconds(engine: OnnxInflectEngine) -> float:
    """Compute a read-ahead target (seconds of buffered audio) from the
    last measured synthesis speed for this engine (same number the
    "Synthesis speed" sensor shows), instead of a fixed config value. A
    comfortably fast engine (high realtime_factor) needs little to no
    read-ahead; one running close to (or slower than) real-time needs
    more slack to absorb timing variance. Falls back to a conservative
    default when there's no prior measurement yet (e.g. the very first
    stream after startup).
    """
    realtime_factor = (engine.last_stats or {}).get("realtime_factor")
    if not realtime_factor or realtime_factor <= 0:
        return _AUTO_READ_AHEAD_TARGET_SECONDS
    return max(
        0.5,
        min(
            MAX_STREAM_READ_AHEAD,
            _AUTO_READ_AHEAD_TARGET_SECONDS / realtime_factor,
        ),
    )


def stats_signal(entry_id: str) -> str:
    """Dispatcher signal name carrying this entry's last-synthesis stats."""
    return f"{DOMAIN}_{entry_id}_stats"

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Inflect TTS speech component via config entry."""
    async_add_entities([InflectTTSEntity(hass, config_entry)])

    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(
        SERVICE_LOAD_MODEL, {}, "async_load_model"
    )
    platform.async_register_entity_service(
        SERVICE_UNLOAD_MODEL, {}, "async_unload_model"
    )


class InflectTTSEntity(TextToSpeechEntity):
    """The Inflect TTS entity."""

    _attr_supported_languages = SUPPORT_LANGUAGES
    _attr_default_language = DEFAULT_LANG
    _attr_supported_options = [CONF_SPEED, CONF_VARIATION, CONF_SEED]

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize Inflect TTS entity."""
        self._hass = hass
        # Options (set via the options flow) override the initial setup
        # data, so changing speed/variation/seed/idle-unload later doesn't
        # require deleting and re-adding the integration.
        settings = {**config_entry.data, **config_entry.options}
        self._model_key = config_entry.data[CONF_MODEL]
        self._default_speed = float(settings.get(CONF_SPEED, DEFAULT_SPEED))
        self._default_variation = float(
            settings.get(CONF_VARIATION, DEFAULT_VARIATION)
        )
        # The number selector always returns floats, but a float seed
        # crashes numpy's RandomState -- cast explicitly.
        self._default_seed = int(settings.get(CONF_SEED, DEFAULT_SEED))
        self._idle_unload_minutes = int(
            settings.get(CONF_IDLE_UNLOAD_MINUTES, DEFAULT_IDLE_UNLOAD_MINUTES)
        )
        self._streaming = bool(settings.get(CONF_STREAMING, DEFAULT_STREAMING))
        # 0 = auto (compute seconds of read-ahead from last measured
        # synthesis speed each stream); any other value is a fixed
        # target in seconds of buffered audio.
        self._stream_read_ahead_seconds = max(
            0.0,
            float(
                settings.get(CONF_STREAM_READ_AHEAD, DEFAULT_STREAM_READ_AHEAD)
            ),
        )
        self._turbo_mode = bool(settings.get(CONF_TURBO_MODE, DEFAULT_TURBO_MODE))
        self._unload_timer_cancel = None

        self._attr_name = MODEL_NAMES[self._model_key]
        self._attr_unique_id = config_entry.entry_id
        self._attr_device_info = DeviceInfo(
            entry_type=DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, config_entry.entry_id)},
            model=MODEL_NAMES[self._model_key],
            name=MODEL_NAMES[self._model_key],
        )
        self._entry_id = config_entry.entry_id

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Run in-process synthesis. Blocking work goes through the
        executor so the event loop never stalls on it."""
        speed = float(options.get(CONF_SPEED, self._default_speed))
        variation = float(options.get(CONF_VARIATION, self._default_variation))
        seed = int(options.get(CONF_SEED, self._default_seed))

        try:
            data, stats = await self._hass.async_add_executor_job(
                synthesize_with_stats,
                self._model_key,
                message,
                speed,
                variation,
                seed,
            )
        except InflectModelError as exc:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="synthesis_error",
                translation_placeholders={"error": str(exc)},
            ) from exc

        if stats is not None:
            async_dispatcher_send(self._hass, stats_signal(self._entry_id), stats)
        self._reschedule_idle_unload()
        return "wav", data

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Stream audio sentence-by-sentence as it's generated, instead
        of waiting for the whole message to finish synthesizing before
        returning anything -- same per-sentence chunking approach used
        for streaming in the companion web app.

        HA always calls this (there's no per-request choice for the end
        user), so the "Streaming synthesis" option is the escape hatch:
        when disabled, fall back to the same buffered behavior as
        async_get_tts_audio, just wrapped as a single-item stream.
        """
        message = "".join([chunk async for chunk in request.message_gen])

        if not self._streaming:
            extension, data = await self.async_get_tts_audio(
                message, request.language, request.options
            )

            async def single_chunk():
                yield data

            return TTSAudioResponse(extension, single_chunk())

        speed = float(request.options.get(CONF_SPEED, self._default_speed))
        variation = float(
            request.options.get(CONF_VARIATION, self._default_variation)
        )
        seed = int(request.options.get(CONF_SEED, self._default_seed))

        try:
            engine, chunk_gen, loaded_fresh = await self._hass.async_add_executor_job(
                get_stream,
                self._model_key,
                message,
                speed,
                variation,
                seed,
                self._turbo_mode,
            )
        except InflectModelError as exc:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="synthesis_error",
                translation_placeholders={"error": str(exc)},
            ) from exc

        async def data_gen():
            # Read-ahead buffer: a background thread drains chunk_gen
            # continuously (not just when we ask for the next chunk), so
            # synthesis of sentence N+1 overlaps with us handing sentence
            # N off to HA/the network, instead of only starting once
            # we've finished transmitting the current chunk. On hardware
            # running close to real-time, that overlap is what prevents
            # an audible gap between sentences.
            read_ahead_seconds = (
                self._stream_read_ahead_seconds or _auto_read_ahead_seconds(engine)
            )
            ahead = _AheadTracker(read_ahead_seconds)
            out_queue: sync_queue.Queue = sync_queue.Queue()
            self._hass.async_add_executor_job(
                _drain_chunks_into_queue, chunk_gen, out_queue, ahead, engine.sample_rate
            )
            try:
                yield _streaming_wav_header(engine.sample_rate)
                while True:
                    item = await self._hass.async_add_executor_job(out_queue.get)
                    if item is _STREAM_DONE:
                        break
                    if isinstance(item, InflectModelError):
                        raise HomeAssistantError(
                            translation_domain=DOMAIN,
                            translation_key="synthesis_error",
                            translation_placeholders={"error": str(item)},
                        ) from item
                    chunk, duration = item
                    yield chunk
                    ahead.remove(duration)
            finally:
                # Runs whether the stream finished, errored, or the
                # consumer disconnected early -- same bookkeeping the
                # non-streaming path does after a synthesis call.
                if engine.last_stats is not None:
                    stats = dict(engine.last_stats)
                    stats["cold_start_seconds"] = (
                        engine.last_load_seconds if loaded_fresh else 0.0
                    )
                    async_dispatcher_send(
                        self._hass, stats_signal(self._entry_id), stats
                    )
                self._reschedule_idle_unload()

        return TTSAudioResponse("wav", data_gen())

    async def async_load_model(self) -> None:
        """Load the ONNX sessions now, ahead of the first TTS request.
        Does not start the idle-unload timer -- the model stays resident
        until a synthesis happens (which then governs it normally) or
        async_unload_model is called explicitly."""
        await self._hass.async_add_executor_job(get_engine, self._model_key)

    async def async_unload_model(self) -> None:
        """Free the ONNX sessions immediately, without waiting for the
        idle-unload timeout."""
        if self._unload_timer_cancel is not None:
            self._unload_timer_cancel()
            self._unload_timer_cancel = None
        await self._hass.async_add_executor_job(unload_engine, self._model_key)

    def _reschedule_idle_unload(self) -> None:
        """(Re)start the idle-unload timer so the model isn't held in
        memory indefinitely on low-end hardware. Disabled when the option
        is set to 0."""
        if self._unload_timer_cancel is not None:
            self._unload_timer_cancel()
            self._unload_timer_cancel = None
        if not self._idle_unload_minutes:
            return

        async def _unload(_now: Any) -> None:
            self._unload_timer_cancel = None
            await self._hass.async_add_executor_job(unload_engine, self._model_key)
            _LOGGER.debug(
                "Unloaded idle Inflect TTS model %s after %d minute(s)",
                self._model_key,
                self._idle_unload_minutes,
            )

        self._unload_timer_cancel = async_call_later(
            self._hass, self._idle_unload_minutes * 60, _unload
        )

    async def async_will_remove_from_hass(self) -> None:
        """Cancel any pending idle-unload timer on teardown."""
        if self._unload_timer_cancel is not None:
            self._unload_timer_cancel()
            self._unload_timer_cancel = None
