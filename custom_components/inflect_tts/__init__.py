"""The Inflect TTS integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import CONF_MODEL
from .model import unload_engine
from .onnxruntime_install import OnnxRuntimeInstallError, ensure_onnxruntime

PLATFORMS = [Platform.TTS, Platform.SENSOR]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Inflect TTS from a config entry."""
    try:
        await hass.async_add_executor_job(ensure_onnxruntime)
    except OnnxRuntimeInstallError as exc:
        # Transient (network, index down) -- let HA retry setup later
        # instead of failing the entry permanently.
        raise ConfigEntryNotReady(str(exc)) from exc

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the entry when its options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload an Inflect TTS config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await hass.async_add_executor_job(unload_engine, entry.data[CONF_MODEL])
    return unloaded
