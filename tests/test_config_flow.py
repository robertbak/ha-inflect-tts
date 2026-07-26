"""Tests for the Inflect TTS config flow (in-process ONNX architecture --
the model artifacts ship inside the integration, no sidecar to reach)."""

from __future__ import annotations

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.inflect_tts.const import (
    CONF_MODEL,
    CONF_SEED,
    CONF_SPEED,
    CONF_STREAM_READ_AHEAD,
    CONF_STREAMING,
    CONF_VARIATION,
    DOMAIN,
    MODEL_MICRO,
    MODEL_NANO,
)


async def test_user_flow_creates_entry(hass: HomeAssistant) -> None:
    """A full user flow with defaults creates a config entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: MODEL_NANO}
    )

    assert result2["type"] is FlowResultType.CREATE_ENTRY
    assert result2["title"] == "Inflect Nano v2"
    assert result2["data"][CONF_MODEL] == MODEL_NANO
    assert result2["data"][CONF_SPEED] == 1.0
    assert result2["data"][CONF_VARIATION] == 0.667
    assert result2["data"][CONF_SEED] == 7
    assert result2["data"][CONF_STREAMING] is True
    assert result2["data"][CONF_STREAM_READ_AHEAD] == 0  # auto


async def test_user_flow_duplicate_model_aborts(hass: HomeAssistant) -> None:
    """Configuring the same model twice aborts the second flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: MODEL_NANO}
    )

    result2 = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result3 = await hass.config_entries.flow.async_configure(
        result2["flow_id"], {CONF_MODEL: MODEL_NANO}
    )

    assert result3["type"] is FlowResultType.ABORT
    assert result3["reason"] == "already_configured"


async def test_user_flow_different_models_both_succeed(hass: HomeAssistant) -> None:
    """Nano and Micro are independent config entries -- no collision."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: MODEL_NANO}
    )
    assert result2["type"] is FlowResultType.CREATE_ENTRY

    result3 = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result4 = await hass.config_entries.flow.async_configure(
        result3["flow_id"], {CONF_MODEL: MODEL_MICRO}
    )
    assert result4["type"] is FlowResultType.CREATE_ENTRY


async def test_user_flow_missing_artifacts_shows_error(
    hass: HomeAssistant, monkeypatch
) -> None:
    """If a model's ONNX files aren't present, the form re-shows with
    an error instead of creating a broken entry."""
    from custom_components.inflect_tts import config_flow

    monkeypatch.setattr(
        config_flow, "MODELS_DIR", config_flow.MODELS_DIR / "does-not-exist"
    )

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: MODEL_NANO}
    )

    assert result2["type"] is FlowResultType.FORM
    assert result2["errors"]["base"] == "model_artifacts_missing"


async def test_options_flow_updates_entry(hass: HomeAssistant) -> None:
    """The options flow (not the model) can be changed after setup."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: MODEL_NANO}
    )
    entry = hass.config_entries.async_get_entry(result2["result"].entry_id)

    options_result = await hass.config_entries.options.async_init(entry.entry_id)
    assert options_result["type"] is FlowResultType.FORM
    assert options_result["step_id"] == "init"

    options_result2 = await hass.config_entries.options.async_configure(
        options_result["flow_id"],
        {
            CONF_SPEED: 1.5,
            CONF_VARIATION: 0.5,
            CONF_SEED: 42,
            "idle_unload_minutes": 5,
            CONF_STREAMING: False,
            CONF_STREAM_READ_AHEAD: 2.0,
        },
    )
    assert options_result2["type"] is FlowResultType.CREATE_ENTRY
    assert options_result2["data"][CONF_SPEED] == 1.5
    assert options_result2["data"][CONF_STREAMING] is False
