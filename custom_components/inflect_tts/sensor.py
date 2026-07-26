"""Exposes the last synthesis's speed (realtime factor) as a sensor, so
it's visible in the UI/history/dashboards instead of only in the logs."""

from __future__ import annotations

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CONF_MODEL, DOMAIN, MODEL_NAMES
from .tts import stats_signal


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the synthesis speed sensor via config entry."""
    async_add_entities([InflectTTSSpeedSensor(config_entry)])


class InflectTTSSpeedSensor(SensorEntity):
    """Realtime factor (Nx) of the most recent synthesis."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = "x realtime"
    _attr_icon = "mdi:speedometer"
    _attr_should_poll = False

    def __init__(self, config_entry: ConfigEntry) -> None:
        model_key = config_entry.data[CONF_MODEL]
        self._entry_id = config_entry.entry_id
        self._attr_name = f"{MODEL_NAMES[model_key]} synthesis speed"
        self._attr_unique_id = f"{config_entry.entry_id}_synthesis_speed"
        self._attr_device_info = DeviceInfo(
            entry_type=DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, config_entry.entry_id)},
            model=MODEL_NAMES[model_key],
            name=MODEL_NAMES[model_key],
        )
        self._attr_native_value = None
        self._attr_extra_state_attributes = {}

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass, stats_signal(self._entry_id), self._handle_stats
            )
        )

    @callback
    def _handle_stats(self, stats: dict) -> None:
        self._attr_native_value = stats["realtime_factor"]
        self._attr_extra_state_attributes = {
            "audio_seconds": stats["audio_seconds"],
            "synthesis_seconds": stats["synthesis_seconds"],
            "rtf": stats["rtf"],
            "characters": stats["characters"],
        }
        self.async_write_ha_state()
