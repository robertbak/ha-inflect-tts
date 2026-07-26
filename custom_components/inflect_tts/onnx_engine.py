"""Torch-free runtime engine: runs the exported ONNX graphs for an Inflect
model. Text frontend + pre/post-processing logic is copied from the
model's own inference.py, which is pure Python/numpy outside of the model
forward pass itself.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import os
import re
import sys
import time
import types
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .const import MODEL_REPOS

_LOGGER = logging.getLogger(__name__)


def _stub_out_unused_segments_backend() -> None:
    """phonemizer.backend eagerly does `import segments` for its
    SegmentsBackend, which we never use (only EspeakBackend). The real
    `segments` package pulls in csvw -> babel/rdflib/jsonschema/... (~50MB)
    purely to satisfy that unused import. phonemizer/backend/segments.py
    lacks `from __future__ import annotations`, so its method signatures
    (e.g. `-> segments.Profile`) are evaluated eagerly at class-definition
    time, not just when SegmentsBackend is instantiated -- the stub needs
    placeholder attributes for that, even though nothing here ever calls
    SegmentsBackend.
    """
    if "segments" not in sys.modules:
        stub = types.ModuleType("segments")
        stub.Tokenizer = object
        stub.Profile = object
        stub.__version__ = "0.0.0-stub"
        sys.modules["segments"] = stub


_stub_out_unused_segments_backend()


class InflectModelError(Exception):
    """Raised when the ONNX model cannot be loaded or run."""


def split_text(text: str, limit: int = 280) -> list[str]:
    normalized = " ".join(text.split())
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?;:])\s+", normalized)
        if part.strip()
    ]
    chunks: list[str] = []
    for sentence in sentences or [normalized]:
        while len(sentence) > limit:
            search = sentence[: limit + 1]
            punctuation = max(search.rfind(mark) for mark in (",", ";", ":"))
            split_at = (
                punctuation + 1
                if punctuation >= limit // 2
                else sentence.rfind(" ", 0, limit + 1)
            )
            if split_at < limit // 2:
                split_at = limit
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def boundary_pause_seconds(chunk: str) -> float:
    ending = chunk.rstrip()[-1:] if chunk.strip() else ""
    return {
        "?": 0.28,
        "!": 0.24,
        ".": 0.22,
        ";": 0.16,
        ":": 0.13,
        ",": 0.09,
    }.get(ending, 0.08)


def edge_fade(waveform: np.ndarray, sample_rate: int, milliseconds: float = 5.0) -> np.ndarray:
    frames = min(round(sample_rate * milliseconds / 1000.0), waveform.size // 2)
    if frames <= 0:
        return waveform
    output = waveform.copy()
    ramp = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float32)
    output[:frames] *= ramp
    output[-frames:] *= ramp[::-1]
    return output


def _intersperse(seq: list[int], item: int) -> list[int]:
    result = [item] * (len(seq) * 2 + 1)
    result[1::2] = seq
    return result


def _import_frontend(artifact_dir: Path) -> types.ModuleType:
    runtime_root = artifact_dir / "runtime"
    for path in (str(runtime_root), str(artifact_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)

    spec = importlib.util.spec_from_file_location(
        "inflect_onnx_frontend", str(artifact_dir / "inflect_vits_frontend.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OnnxInflectEngine:
    """Loads and runs the two-graph ONNX export of an Inflect model."""

    def __init__(self, model_key: str, artifact_root: str) -> None:
        if model_key not in MODEL_REPOS:
            raise InflectModelError(f"Unknown Inflect model '{model_key}'")
        self._model_key = model_key
        self._artifact_dir = Path(artifact_root) / model_key
        self._duration_sess: ort.InferenceSession | None = None
        self._decode_sess: ort.InferenceSession | None = None
        self._frontend = None
        self._sample_rate = 24000
        self._add_blank = True
        self.last_stats: dict[str, float | int | str] | None = None

    @property
    def is_loaded(self) -> bool:
        return self._duration_sess is not None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load(self) -> None:
        if not self._artifact_dir.exists():
            raise InflectModelError(
                f"No exported ONNX artifacts found at {self._artifact_dir}. "
                "This image must be built with the export stage for this model."
            )
        try:
            config = json.loads((self._artifact_dir / "config.json").read_text())
            self._sample_rate = int(config["data"]["sampling_rate"])
            self._add_blank = bool(config["data"]["add_blank"])

            self._frontend = _import_frontend(self._artifact_dir)

            sess_options = ort.SessionOptions()
            # Default ORT spins up a thread pool sized to the core count
            # for *each* session, plus a growing memory arena -- fine on a
            # dev workstation but enough to OOM a Pi-class SBC once both
            # sessions and HA's own threads are accounted for. These are
            # small/fast graphs, so a couple of threads costs little speed.
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_cpu_mem_arena = False
            sess_options.enable_mem_pattern = False
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )

            self._duration_sess = ort.InferenceSession(
                str(self._artifact_dir / "duration.onnx"),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._decode_sess = ort.InferenceSession(
                str(self._artifact_dir / "decode.onnx"),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001
            raise InflectModelError(f"Failed to load ONNX model {self._model_key}: {exc}") from exc

    def _tokens(self, text: str) -> np.ndarray:
        from text import cleaned_text_to_sequence  # noqa: E402 (frontend inserted onto sys.path)

        phonemes = self._frontend.run_vits_frontend(text).phoneme_text
        sequence = cleaned_text_to_sequence(phonemes)
        if self._add_blank:
            sequence = _intersperse(sequence, 0)
        if not sequence:
            raise InflectModelError("The text frontend produced no speakable tokens.")
        return np.asarray([sequence], dtype=np.int64)

    def _prepare(self, text: str) -> tuple[str, list[str]]:
        normalized = " ".join(text.split())
        if not normalized:
            raise InflectModelError("Text must not be empty.")
        if normalized[-1] not in ".!?;:":
            # Without a closing punctuation cue the duration predictor has
            # no signal that the utterance is ending, so the last word/
            # syllable's audio gets allocated too few frames and sounds
            # clipped. This is audio-only -- doesn't affect the caller's text.
            normalized += "."
        return normalized, split_text(normalized)

    def _run_chunk(
        self, chunk: str, speed: float, variation: float, rng: np.random.RandomState
    ) -> np.ndarray:
        """Run one sentence chunk through both graphs. Returns a
        float32 waveform piece (not yet clipped/quantized)."""
        tokens = self._tokens(chunk)
        lengths = np.asarray([tokens.shape[1]], dtype=np.int64)
        length_scale = np.asarray(1.0 / speed, dtype=np.float32)

        m_p_exp, logs_p_exp, y_mask = self._duration_sess.run(
            None,
            {"tokens": tokens, "lengths": lengths, "length_scale": length_scale},
        )

        zp_noise = rng.standard_normal(m_p_exp.shape).astype(np.float32)
        noise_scale = np.asarray(variation, dtype=np.float32)

        (waveform,) = self._decode_sess.run(
            None,
            {
                "m_p_exp": m_p_exp,
                "logs_p_exp": logs_p_exp,
                "y_mask": y_mask,
                "zp_noise": zp_noise,
                "noise_scale": noise_scale,
            },
        )
        return edge_fade(waveform[0, 0].astype(np.float32), self._sample_rate)

    def _pause_piece(self, previous_chunk: str) -> np.ndarray:
        return np.zeros(
            round(self._sample_rate * boundary_pause_seconds(previous_chunk)),
            dtype=np.float32,
        )

    @staticmethod
    def _pcm16_bytes(piece: np.ndarray) -> bytes:
        return (np.clip(piece, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    def _record_stats(
        self, normalized: str, total_samples: int, start_time: float
    ) -> None:
        elapsed = time.monotonic() - start_time
        audio_seconds = total_samples / self._sample_rate
        rtf = elapsed / audio_seconds if audio_seconds else float("inf")
        realtime_factor = round(1 / rtf) if rtf else 0
        self.last_stats = {
            "audio_seconds": round(audio_seconds, 2),
            "synthesis_seconds": round(elapsed, 2),
            "rtf": round(rtf, 3),
            "realtime_factor": realtime_factor,
            "characters": len(normalized),
            "model": self._model_key,
        }
        _LOGGER.info(
            "Synthesized %.1fs of audio in %.2fs (RTF %.2f, %dx realtime) "
            "for %d chars, model=%s",
            audio_seconds,
            elapsed,
            rtf,
            realtime_factor,
            len(normalized),
            self._model_key,
        )

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 7,
    ) -> bytes:
        if self._duration_sess is None or self._decode_sess is None:
            raise InflectModelError("Model is not loaded")

        # np.random.RandomState requires an int seed -- callers (e.g. HA's
        # NumberSelector-backed config options) may hand us a float.
        speed = float(speed)
        variation = float(variation)
        seed = int(seed)

        normalized, chunks = self._prepare(text)
        start_time = time.monotonic()
        pieces: list[np.ndarray] = []
        rng = np.random.RandomState(seed)

        try:
            for index, chunk in enumerate(chunks):
                if index:
                    pieces.append(self._pause_piece(chunks[index - 1]))
                pieces.append(self._run_chunk(chunk, speed, variation, rng))
        except InflectModelError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InflectModelError(f"Synthesis failed: {exc}") from exc

        full = np.concatenate(pieces)
        pcm16 = self._pcm16_bytes(full)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(pcm16)

        self._record_stats(normalized, full.size, start_time)
        return buffer.getvalue()

    def synthesize_stream(
        self,
        text: str,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 7,
    ):
        """Like synthesize(), but yields raw 16-bit PCM bytes per sentence
        chunk (plus inter-sentence pause silence) as each is generated,
        instead of building the whole utterance before returning anything.
        No WAV container -- the caller (tts.py's streaming path) wraps a
        single streaming-safe header around the whole sequence.

        Blocking, like synthesize() -- each next() call runs model
        inference, so callers must pull this via the executor.
        """
        if self._duration_sess is None or self._decode_sess is None:
            raise InflectModelError("Model is not loaded")

        speed = float(speed)
        variation = float(variation)
        seed = int(seed)

        normalized, chunks = self._prepare(text)
        start_time = time.monotonic()
        rng = np.random.RandomState(seed)
        total_samples = 0

        try:
            for index, chunk in enumerate(chunks):
                if index:
                    pause = self._pause_piece(chunks[index - 1])
                    total_samples += pause.size
                    yield self._pcm16_bytes(pause)
                piece = self._run_chunk(chunk, speed, variation, rng)
                total_samples += piece.size
                yield self._pcm16_bytes(piece)
        except InflectModelError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InflectModelError(f"Synthesis failed: {exc}") from exc

        self._record_stats(normalized, total_samples, start_time)
