"""Adapters for model packages. Importing hotword never downloads model weights."""

from __future__ import annotations

from importlib import resources
import os
from pathlib import Path
import shutil
from threading import RLock
import tempfile
from typing import Any, Callable, Sequence
from zipfile import ZipFile

import numpy as np


_cactus_lock = RLock()  # Whistle and Needle share a process-wide native engine.


def _cache_copy(source: str, destination: Path, *, member: str | None = None) -> None:
    """Put a complete downloaded asset where the Cactus SDK expects it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            if member is None:
                with open(source, "rb") as original:
                    shutil.copyfileobj(original, temporary)
            else:
                with ZipFile(source) as archive, archive.open(member) as original:
                    shutil.copyfileobj(original, temporary)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, destination)


def _prepare_whistle_assets() -> str:
    """Fetch weights and engine without the SDK's forced config.json request."""
    from huggingface_hub import hf_hub_download
    from needle.agent import fetch

    version = 3
    engine = Path(fetch.cache_dir(version)) / fetch.lib_name(version)
    if not engine.is_file():
        tag = fetch._platform_tag()
        wheel = hf_hub_download(
            repo_id=fetch.engine_repo(version),
            filename=fetch.engine_wheel(fetch.engine_version(version), tag, version),
            repo_type="model",
        )
        name = fetch.lib_name(version, tag)
        stem, suffix = os.path.splitext(name)
        _cache_copy(wheel, engine, member=f"needle/{stem}{version}{suffix}")

    weights = Path(fetch.cache_dir(fetch.WHISTLE)) / fetch.base_weights(fetch.WHISTLE)
    if not weights.is_file():
        downloaded = hf_hub_download(
            repo_id=fetch.engine_repo(fetch.WHISTLE),
            filename=fetch.base_weights(fetch.WHISTLE),
            repo_type="model",
        )
        _cache_copy(downloaded, weights)
    return str(weights)


class SileroVAD:
    """Silero v6.2.3 ONNX VAD at 16 kHz, one 512-sample frame at a time."""

    def __init__(self) -> None:
        self._model: Any = None
        self.reset()

    def warmup(self) -> None:
        if self._model is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            model = resources.files("hotword").joinpath("data", "silero_vad.onnx").read_bytes()
            self._model = ort.InferenceSession(
                model,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

    def probability(self, frame: np.ndarray) -> float:
        if frame.shape != (512,) or frame.dtype != np.float32:
            raise ValueError("SileroVAD needs 512 float32 samples at 16 kHz")
        self.warmup()
        audio = np.concatenate((self._context, frame.reshape(1, 512)), axis=1)
        output, state = self._model.run(
            None,
            {"input": audio, "state": self._state, "sr": np.array(16000, dtype=np.int64)},
        )
        self._state = state
        self._context = audio[:, -64:].copy()
        return float(output[0, 0])

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)


class WhistleTranscriber:
    """Cactus Whistle speech recognition through the cactus-needle package."""

    def __init__(
        self,
        *,
        keywords: Sequence[str] = (),
        language: str | None = None,
        weights: str | None = None,
    ) -> None:
        self.keywords = list(keywords)
        self.language = language
        self.weights = weights
        self._model: Any = None

    def transcribe(self, samples: np.ndarray) -> str:
        if len(samples) > 30 * 16000:
            raise ValueError("Whistle accepts at most 30 seconds of audio")
        with _cactus_lock:
            self.warmup()
            result = self._model.transcribe(
                samples, language=self.language, keywords=self.keywords
            )
        return str(result["text"])

    def warmup(self) -> None:
        with _cactus_lock:
            if self._model is None:
                from needle import Whistle

                weights = self.weights or os.environ.get("NEEDLE_WHISTLE_WEIGHTS")
                if weights is None:
                    weights = _prepare_whistle_assets()
                self._model = Whistle(weights=weights)


class WhisperTranscriber:
    """Optional faster-whisper backend. Install with uv sync --extra whisper."""

    def __init__(self, model: str = "tiny", *, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self._model: Any = None

    def transcribe(self, samples: np.ndarray) -> str:
        self.warmup()
        segments, _ = self._model.transcribe(samples, vad_filter=False)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def warmup(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model, device=self.device)


class NeedleRouter:
    """Run registered Python functions for speech after a wakeword.

    Each function must be decorated with ``@needle.tool``. A fresh Needle
    conversation starts for every command, so previous speech cannot leak in.
    """

    def __init__(self, tools: Sequence[Callable[..., Any]], *, weights: str | None = None) -> None:
        if not tools:
            raise ValueError("NeedleRouter needs at least one tool")
        self.tools = list(tools)
        self.weights = weights
        self._agent: Any = None

    def run(self, command: str) -> dict[str, Any]:
        with _cactus_lock:
            self.warmup()
            self._agent.reset()
            return self._agent.run(command)

    def warmup(self) -> None:
        with _cactus_lock:
            if self._agent is None:
                from needle import Needle

                self._agent = Needle(tools=self.tools, weights=self.weights)

    def close(self) -> None:
        with _cactus_lock:
            if self._agent is not None:
                self._agent.close()
                self._agent = None
