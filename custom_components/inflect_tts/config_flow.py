"""Config flow for the Inflect TTS integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import (
    CONF_IDLE_UNLOAD_MINUTES,
    CONF_MODEL,
    CONF_SEED,
    CONF_SPEED,
    CONF_VARIATION,
    DEFAULT_IDLE_UNLOAD_MINUTES,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DEFAULT_SPEED,
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

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_MODEL, default=DEFAULT_MODEL): vol.In(MODEL_NAMES),
        vol.Optional(CONF_SPEED, default=DEFAULT_SPEED): vol.All(
            vol.Coerce(float), vol.Range(min=MIN_SPEED, max=MAX_SPEED)
        ),
        vol.Optional(CONF_VARIATION, default=DEFAULT_VARIATION): vol.All(
            vol.Coerce(float), vol.Range(min=MIN_VARIATION, max=MAX_VARIATION)
        ),
        vol.Optional(CONF_SEED, default=DEFAULT_SEED): vol.Coerce(int),
        vol.Optional(
            CONF_IDLE_UNLOAD_MINUTES, default=DEFAULT_IDLE_UNLOAD_MINUTES
        ): vol.All(
            vol.Coerce(int),
            vol.Range(min=MIN_IDLE_UNLOAD_MINUTES, max=MAX_IDLE_UNLOAD_MINUTES),
        ),
    }
)


class InflectTTSConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Inflect TTS."""

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
