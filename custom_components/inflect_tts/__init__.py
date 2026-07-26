"""The Inflect TTS integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import CONF_MODEL
from .model import unload_engine
from .onnxruntime_install import ensure_onnxruntime

PLATFORMS = [Platform.TTS]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Inflect TTS from a config entry."""
    await hass.async_add_executor_job(ensure_onnxruntime)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload an Inflect TTS config entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await hass.async_add_executor_job(unload_engine, entry.data[CONF_MODEL])
    return unloaded
