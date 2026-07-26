"""Ensures onnxruntime is installed.

manifest.json's `requirements` can only pull from PyPI (or HA's own
musllinux wheel mirror), and onnxruntime isn't published for musllinux
there -- so it can't be listed as a normal requirement. Instead, install
it here from our own PEP 503 index (see const.ONNXRUNTIME_INDEX_URL),
which hosts wheels for every architecture HA runs on; pip resolves the
right one for whatever machine this is automatically.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from .const import ONNXRUNTIME_INDEX_URL, ONNXRUNTIME_VERSION

_LOGGER = logging.getLogger(__name__)


class OnnxRuntimeInstallError(Exception):
    """Raised when onnxruntime can't be installed."""


def ensure_onnxruntime() -> None:
    """Install onnxruntime if it isn't already. Blocking -- call via
    hass.async_add_executor_job, never directly from the event loop."""
    try:
        import onnxruntime  # noqa: F401

        return
    except ImportError:
        pass

    _LOGGER.info(
        "onnxruntime not found, installing %s from %s",
        ONNXRUNTIME_VERSION,
        ONNXRUNTIME_INDEX_URL,
    )
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--extra-index-url",
                ONNXRUNTIME_INDEX_URL,
                f"onnxruntime=={ONNXRUNTIME_VERSION}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise OnnxRuntimeInstallError(
            "Timed out installing onnxruntime after 5 minutes -- check "
            "network connectivity to " + ONNXRUNTIME_INDEX_URL
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise OnnxRuntimeInstallError(
            f"pip install failed (exit {exc.returncode}): "
            f"{exc.stderr.strip() if exc.stderr else 'no output'}"
        ) from exc
