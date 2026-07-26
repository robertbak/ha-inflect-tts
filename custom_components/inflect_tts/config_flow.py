"""Config flow for the Inflect TTS integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_IDLE_UNLOAD_MINUTES,
    CONF_MODEL,
    CONF_SEED,
    CONF_SPEED,
    CONF_STREAMING,
    CONF_VARIATION,
    DEFAULT_IDLE_UNLOAD_MINUTES,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DEFAULT_SPEED,
    DEFAULT_STREAMING,
    DEFAULT_VARIATION,
    DOMAIN,
    MAX_IDLE_UNLOAD_MINUTES,
    MAX_SPEED,
    MAX_VARIATION,
    MIN_IDLE_UNLOAD_MINUTES,
    MIN_SPEED,
    MIN_VARIATION,
    MODEL_NAMES,
    MODELS_DIR,
)

_LOGGER = logging.getLogger(__name__)

_SPEED_SELECTOR = selector.NumberSelector(
    selector.NumberSelectorConfig(
        min=MIN_SPEED, max=MAX_SPEED, step=0.05, mode=selector.NumberSelectorMode.BOX
    )
)
_VARIATION_SELECTOR = selector.NumberSelector(
    selector.NumberSelectorConfig(
        min=MIN_VARIATION,
        max=MAX_VARIATION,
        step=0.01,
        mode=selector.NumberSelectorMode.BOX,
    )
)
_SEED_SELECTOR = selector.NumberSelector(
    selector.NumberSelectorConfig(step=1, mode=selector.NumberSelectorMode.BOX)
)
_IDLE_UNLOAD_SELECTOR = selector.NumberSelector(
    selector.NumberSelectorConfig(
        min=MIN_IDLE_UNLOAD_MINUTES,
        max=MAX_IDLE_UNLOAD_MINUTES,
        step=1,
        mode=selector.NumberSelectorMode.BOX,
    )
)


def _tuning_fields(current: dict[str, Any]) -> dict[Any, Any]:
    """Speed/variation/seed/idle-unload fields, shared between the initial
    setup form and the options form."""
    return {
        vol.Optional(
            CONF_SPEED, default=current.get(CONF_SPEED, DEFAULT_SPEED)
        ): _SPEED_SELECTOR,
        vol.Optional(
            CONF_VARIATION, default=current.get(CONF_VARIATION, DEFAULT_VARIATION)
        ): _VARIATION_SELECTOR,
        vol.Optional(
            CONF_SEED, default=current.get(CONF_SEED, DEFAULT_SEED)
        ): _SEED_SELECTOR,
        vol.Optional(
            CONF_IDLE_UNLOAD_MINUTES,
            default=current.get(
                CONF_IDLE_UNLOAD_MINUTES, DEFAULT_IDLE_UNLOAD_MINUTES
            ),
        ): _IDLE_UNLOAD_SELECTOR,
        vol.Optional(
            CONF_STREAMING, default=current.get(CONF_STREAMING, DEFAULT_STREAMING)
        ): bool,
    }


STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_MODEL, default=DEFAULT_MODEL): vol.In(MODEL_NAMES),
        **_tuning_fields({}),
    }
)


def _options_schema(current: dict[str, Any]) -> vol.Schema:
    return vol.Schema(_tuning_fields(current))


class InflectTTSOptionsFlow(OptionsFlow):
    """Handle options (speed/variation/seed/idle-unload) after setup --
    the model itself isn't editable here since changing it would collide
    with the uniqueness check in the initial config flow."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        current = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(
            step_id="init", data_schema=_options_schema(current)
        )


class InflectTTSConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Inflect TTS."""

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> InflectTTSOptionsFlow:
        return InflectTTSOptionsFlow()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            model_key = user_input[CONF_MODEL]

            # Model artifacts (ONNX graphs + text frontend) ship inside
            # the integration itself now -- no sidecar to reach, just
            # confirm the files this model needs are actually present.
            artifacts_present = await self.hass.async_add_executor_job(
                lambda: (MODELS_DIR / model_key / "duration.onnx").exists()
                and (MODELS_DIR / model_key / "decode.onnx").exists()
            )
            if not artifacts_present:
                errors["base"] = "model_artifacts_missing"

            if not errors:
                self._async_abort_entries_match({CONF_MODEL: model_key})
                title = MODEL_NAMES[model_key]
                return self.async_create_entry(title=title, data=user_input)

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )
