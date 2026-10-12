"""Continuous microphone capture for the wakeword detector."""

from __future__ import annotations

import json
from queue import Empty, Full, Queue
import subprocess
import sys
from threading import Event
from typing import Iterator

import numpy as np

from .core import FRAME_SIZE, SAMPLE_RATE


def _probe_microphone(device: str | int | None) -> None:
    """Check PortAudio in a process we can stop if initialization hangs."""
    program = (
        "import json, sys, sounddevice as sd; "
        "sd.check_input_settings(device=json.loads(sys.argv[1]), "
        "channels=1, dtype='float32', samplerate=16000)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", program, json.dumps(device)],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Microphone initialization timed out. Check that your audio server and microphone are available."
        ) from exc
    if result.returncode != 0:
        reason = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        raise RuntimeError(f"Microphone is unavailable: {reason}")


class MicrophoneStream:
    """Capture float32 audio on a callback thread while models run elsewhere.

    Use as a context manager. If processing falls behind the bounded queue,
    iteration raises instead of silently losing audio.
    """

    def __init__(self, *, device: str | int | None = None, buffer_seconds: float = 10.0) -> None:
        if buffer_seconds <= 0:
            raise ValueError("buffer_seconds must be positive")
        self.device = device
        capacity = max(1, round(buffer_seconds * SAMPLE_RATE / FRAME_SIZE))
        self._chunks: Queue[np.ndarray] = Queue(maxsize=capacity)
        self._overflow = Event()
        self._stream = None

    @property
    def queued_seconds(self) -> float:
        """Approximate amount of captured audio waiting for processing."""
        return self._chunks.qsize() * FRAME_SIZE / SAMPLE_RATE

    def __enter__(self) -> MicrophoneStream:
        _probe_microphone(self.device)
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(f"Could not initialize microphone audio: {exc}") from exc

        def capture(indata, frames, _time, status) -> None:
            if status.input_overflow or frames != FRAME_SIZE:
                self._overflow.set()
            try:
                self._chunks.put_nowait(indata[:, 0].copy())
            except Full:
                self._overflow.set()

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SIZE,
                device=self.device,
                callback=capture,
            )
            self._stream.start()
        except Exception as exc:
            if self._stream is not None:
                self._stream.close()
            self._stream = None
            raise RuntimeError(f"Could not open microphone: {exc}") from exc
        return self

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._stream is None:
            raise RuntimeError("Use MicrophoneStream inside a with block")
        while True:
            if self._overflow.is_set():
                raise RuntimeError("Microphone audio overflowed; processing could not keep up")
            try:
                yield self._chunks.get(timeout=0.2)
            except Empty:
                if self._stream is not None and not self._stream.active:
                    raise RuntimeError("Microphone stopped")

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
