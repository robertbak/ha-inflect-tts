from __future__ import annotations

import re
import tempfile
import threading
from dataclasses import dataclass

import inflect_nano_v2_frontend as _frontend_module
from inflect_nano_v2_frontend import _configure_espeak, normalize_text

# Guards the temporary tempfile.tempdir override below -- HA can run
# multiple synthesis requests concurrently on different executor
# threads, and tempfile.tempdir is global process state, so two
# overlapping calls could otherwise restore each other's value early.
_TEMPDIR_OVERRIDE_LOCK = threading.Lock()


# eSpeak is the general fallback. This table contains verified exceptions only;
# every entry is covered by a regression test and listening audit.
PHONEME_OVERRIDES = {
    "sˈæskɐtʃˌuːən": "sɐskˈætʃəwən",
    "flʊɹɹˈɛsənt": "flʊˈɹɛsənt",
}


@dataclass(frozen=True)
class VitsFrontendOutput:
    raw_text: str
    normalized_text: str
    phoneme_text: str


def phonemize_normalized(normalized_text: str) -> str:
    return phonemize_normalized_batch([normalized_text], jobs=1)[0]


def _apply_phoneme_overrides(phoneme_text: str) -> str:
    for source, replacement in PHONEME_OVERRIDES.items():
        phoneme_text = phoneme_text.replace(source, replacement)
    return re.sub(r"\s+", " ", phoneme_text).strip()


def phonemize_normalized_batch(normalized_texts: list[str], *, jobs: int = 1) -> list[str]:
    if not normalized_texts:
        return []
    _configure_espeak()
    from phonemizer import phonemize

    # Narrow, restore-after-use override: phonemizer's EspeakAPI copies
    # the espeak library into a fresh tempfile.mkdtemp() dir on backend
    # construction, which needs an exec-permitted directory on HAOS (see
    # ESPEAK_TMP_DIR's definition in inflect_nano_v2_frontend.py for why
    # this must NOT be a permanent global override).
    with _TEMPDIR_OVERRIDE_LOCK:
        previous_tempdir = tempfile.tempdir
        if _frontend_module.ESPEAK_TMP_DIR:
            tempfile.tempdir = _frontend_module.ESPEAK_TMP_DIR
        try:
            phoneme_texts = phonemize(
                normalized_texts,
                language="en-us",
                backend="espeak",
                strip=True,
                preserve_punctuation=True,
                with_stress=True,
                njobs=jobs,
            )
        finally:
            tempfile.tempdir = previous_tempdir

    return [_apply_phoneme_overrides(text) for text in phoneme_texts]


def run_vits_frontend_batch(texts: list[str], *, jobs: int = 1) -> list[VitsFrontendOutput]:
    normalized = [normalize_text(text) for text in texts]
    phonemes = phonemize_normalized_batch(normalized, jobs=jobs)
    return [
        VitsFrontendOutput(raw_text=raw, normalized_text=norm, phoneme_text=phones)
        for raw, norm, phones in zip(texts, normalized, phonemes, strict=True)
    ]


def run_vits_frontend(text: str) -> VitsFrontendOutput:
    normalized = normalize_text(text)
    return VitsFrontendOutput(
        raw_text=text,
        normalized_text=normalized,
        phoneme_text=phonemize_normalized(normalized),
    )
