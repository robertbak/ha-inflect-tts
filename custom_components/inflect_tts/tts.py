"""Support for the Inflect local TTS models (Micro/Nano), in-process."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_call_later

from .const import (
    CONF_IDLE_UNLOAD_MINUTES,
    CONF_MODEL,
    CONF_SEED,
    CONF_SPEED,
    CONF_VARIATION,
    DEFAULT_IDLE_UNLOAD_MINUTES,
    DEFAULT_LANG,
    DEFAULT_SEED,
    DEFAULT_SPEED,
    DEFAULT_VARIATION,
    DOMAIN,
    MODEL_NAMES,
    SUPPORT_LANGUAGES,
)
from .model import InflectModelError, synthesize, unload_engine

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Inflect TTS speech component via config entry."""
    async_add_entities([InflectTTSEntity(hass, config_entry)])


class InflectTTSEntity(TextToSpeechEntity):
    """The Inflect TTS entity."""

    _attr_supported_languages = SUPPORT_LANGUAGES
    _attr_default_language = DEFAULT_LANG
    _attr_supported_options = [CONF_SPEED, CONF_VARIATION, CONF_SEED]

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize Inflect TTS entity."""
        self._hass = hass
        self._model_key = config_entry.data[CONF_MODEL]
        self._default_speed = config_entry.data.get(CONF_SPEED, DEFAULT_SPEED)
        self._default_variation = config_entry.data.get(
            CONF_VARIATION, DEFAULT_VARIATION
        )
        self._default_seed = config_entry.data.get(CONF_SEED, DEFAULT_SEED)
        self._idle_unload_minutes = config_entry.data.get(
            CONF_IDLE_UNLOAD_MINUTES, DEFAULT_IDLE_UNLOAD_MINUTES
        )
        self._unload_timer_cancel = None

        self._attr_name = MODEL_NAMES[self._model_key]
        self._attr_unique_id = config_entry.entry_id
        self._attr_device_info = DeviceInfo(
            entry_type=DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, config_entry.entry_id)},
            model=MODEL_NAMES[self._model_key],
            name=MODEL_NAMES[self._model_key],
        )

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Run in-process synthesis. Blocking work goes through the
        executor so the event loop never stalls on it."""
        speed = options.get(CONF_SPEED, self._default_speed)
        variation = options.get(CONF_VARIATION, self._default_variation)
        seed = options.get(CONF_SEED, self._default_seed)

        try:
            data = await self._hass.async_add_executor_job(
                synthesize,
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

        self._reschedule_idle_unload()
        return "wav", data

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
