"""Exposes the last synthesis's speed (realtime factor) as a sensor, so
it's visible in the UI/history/dashboards instead of only in the logs."""

from __future__ import annotations

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTime
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
    """Set up the diagnostic sensors via config entry."""
    async_add_entities(
        [InflectTTSSpeedSensor(config_entry), InflectTTSModelLoadTimeSensor(config_entry)]
    )


class _InflectTTSStatsSensor(SensorEntity):
    """Shared device info/dispatcher wiring for the stats-driven sensors."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_should_poll = False
    _attr_state_class = SensorStateClass.MEASUREMENT

    _name_suffix: str
    _unique_id_suffix: str

    def __init__(self, config_entry: ConfigEntry) -> None:
        model_key = config_entry.data[CONF_MODEL]
        self._entry_id = config_entry.entry_id
        self._attr_name = f"{MODEL_NAMES[model_key]} {self._name_suffix}"
        self._attr_unique_id = f"{config_entry.entry_id}_{self._unique_id_suffix}"
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
        raise NotImplementedError


class InflectTTSSpeedSensor(_InflectTTSStatsSensor):
    """Realtime factor (Nx) of the most recent synthesis."""

    _name_suffix = "synthesis speed"
    _unique_id_suffix = "synthesis_speed"

    _attr_native_unit_of_measurement = "x realtime"
    _attr_icon = "mdi:speedometer"
    _attr_suggested_display_precision = 2

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


class InflectTTSModelLoadTimeSensor(_InflectTTSStatsSensor):
    """How long the most recent request spent loading the ONNX sessions
    before synthesis could even start -- 0 on a warm request (the model
    was already resident), positive right after the idle-unload timer
    (or a fresh HA start) freed it and a new request had to reload it.
    Useful for telling "synthesis is slow" apart from "loading the model
    is what's slow" on constrained hardware."""

    _name_suffix = "model load time"
    # Kept as-is (not renamed to match the class) so existing dashboards/
    # automations referencing this entity by unique_id/entity_id aren't
    # orphaned by what's otherwise just a display-name change.
    _unique_id_suffix = "cold_start_seconds"

    _attr_device_class = SensorDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.SECONDS
    _attr_icon = "mdi:timer-sand"
    _attr_suggested_display_precision = 2

    @callback
    def _handle_stats(self, stats: dict) -> None:
        cold_start = stats.get("cold_start_seconds")
        if cold_start is None:
            return
        self._attr_native_value = cold_start
        self.async_write_ha_state()
