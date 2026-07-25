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
    )
