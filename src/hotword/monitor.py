"""Periodic resource and detector status reporting for live listening."""

from __future__ import annotations

import sys
import time
import os
import resource
from pathlib import Path
from threading import Event, Thread
from typing import TextIO

from .audio import MicrophoneStream
from .core import WakewordDetector


def rss_mib() -> float:
    """Current RSS on Linux; high-water RSS on other Unix systems."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1048576
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class ResourceMonitor:
    def __init__(
        self, detector: WakewordDetector, microphone: MicrophoneStream,
        *, interval: float = 2.0, output: TextIO | None = None,
    ) -> None:
        self.detector = detector
        self.microphone = microphone
        self.interval = interval
        self.output = output or sys.stderr
        self._stop = Event()
        self._thread: Thread | None = None
        self._previous_cpu = time.process_time()
        self._previous_time = time.monotonic()

    def sample(self) -> str:
        now = time.monotonic()
        cpu = time.process_time()
        elapsed = max(now - self._previous_time, 1e-6)
        cpu_percent = 100 * (cpu - self._previous_cpu) / elapsed if elapsed >= 0.1 else 0.0
        self._previous_cpu, self._previous_time = cpu, now
        status = self.detector.status
        memory = rss_mib()
        line = (
            f"[hotword] stage={status.stage} cpu={cpu_percent:.1f}% "
            f"rss={memory:.1f}MiB audio={status.audio_seconds:.1f}s "
            f"queue={self.microphone.queued_seconds:.2f}s "
            f"vad={status.vad_probability:.2f} utterances={status.utterances} "
            f"matches={status.matches} speaker_rejections={status.speaker_rejections} "
            f"last_pipeline={status.last_inference_ms:.0f}ms"
        )
        print(line, file=self.output, flush=True)
        return line

    def __enter__(self) -> ResourceMonitor:
        self.sample()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval)
