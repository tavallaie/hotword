"""Audio framing, wakeword matching, and action dispatch."""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

import numpy as np

from .speaker import SpeakerDecision


SAMPLE_RATE = 16000
FRAME_SIZE = 512


class VoiceActivityDetector(Protocol):
    def probability(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class Transcriber(Protocol):
    def transcribe(self, samples: np.ndarray) -> str: ...


class CommandRouter(Protocol):
    def run(self, command: str) -> Any: ...


class SpeakerVerifier(Protocol):
    def verify(self, samples: np.ndarray) -> SpeakerDecision: ...


@dataclass(frozen=True)
class DetectorStatus:
    stage: str
    audio_seconds: float
    vad_probability: float
    utterances: int
    matches: int
    speaker_rejections: int
    last_inference_ms: float
    last_speaker_similarity: float | None


@dataclass(frozen=True)
class Match:
    wakeword: str
    transcript: str
    command: str
    start_seconds: float
    end_seconds: float
    action_result: Any = None
    speaker: str | None = None
    speaker_similarity: float | None = None


class WakewordDetector:
    """Feed 16 kHz mono float samples; receive matches after speech ends.

    ``on_wake`` runs even when there is no command. ``router`` only receives
    nonempty speech after the wakeword.
    """

    def __init__(
        self,
        wakewords: Sequence[str],
        vad: VoiceActivityDetector,
        transcriber: Transcriber,
        *,
        on_wake: Callable[[Match], Any] | None = None,
        router: CommandRouter | None = None,
        speaker_verifier: SpeakerVerifier | None = None,
        threshold: float = 0.5,
        silence_seconds: float = 0.5,
        max_utterance_seconds: float = 12.0,
        cooldown_seconds: float = 1.0,
    ) -> None:
        if isinstance(wakewords, str):
            wakewords = [wakewords]
        if not wakewords or any(not word.strip() for word in wakewords):
            raise ValueError("Provide at least one nonempty wakeword")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if silence_seconds <= 0 or max_utterance_seconds <= 0 or cooldown_seconds < 0:
            raise ValueError("Durations must be positive; cooldown may be zero")
        if max_utterance_seconds > 29:
            raise ValueError("Utterances must be at most 29 seconds for Whistle")
        self.wakewords = tuple(wakewords)
        self.vad = vad
        self.transcriber = transcriber
        self.on_wake = on_wake
        self.router = router
        self.speaker_verifier = speaker_verifier
        self.threshold = threshold
        self.silence_samples = round(silence_seconds * SAMPLE_RATE)
        self.max_samples = round(max_utterance_seconds * SAMPLE_RATE)
        self.cooldown_seconds = cooldown_seconds
        self._patterns = [
            re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, word.split())) + r"(?!\w)", re.I)
            for word in self.wakewords
        ]
        self.reset()

    def reset(self) -> None:
        """Begin a new audio stream and clear VAD and cooldown state."""
        self.vad.reset()
        self._pending = np.empty(0, dtype=np.float32)
        self._preroll: deque[np.ndarray] = deque(maxlen=8)
        self._active: list[np.ndarray] = []
        self._start_sample = 0
        self._processed = 0
        self._silence = 0
        self._last_match: dict[str, float] = {}
        self._stage = "idle"
        self._last_vad_probability = 0.0
        self._utterances = 0
        self._matches = 0
        self._speaker_rejections = 0
        self._last_inference_ms = 0.0
        self._last_speaker_similarity: float | None = None

    @property
    def status(self) -> DetectorStatus:
        """A snapshot for monitoring a live detector."""
        return DetectorStatus(
            stage=self._stage,
            audio_seconds=self._processed / SAMPLE_RATE,
            vad_probability=self._last_vad_probability,
            utterances=self._utterances,
            matches=self._matches,
            speaker_rejections=self._speaker_rejections,
            last_inference_ms=self._last_inference_ms,
            last_speaker_similarity=self._last_speaker_similarity,
        )

    def warmup(self) -> None:
        """Load configured models before opening a live audio source."""
        for backend in (self.vad, self.transcriber, self.router, self.speaker_verifier):
            prepare = getattr(backend, "warmup", None)
            if prepare is not None:
                prepare()

    def feed(self, samples: np.ndarray) -> list[Match]:
        """Consume a chunk of normalized 16 kHz mono float audio."""
        chunk = np.asarray(samples)
        if chunk.ndim != 1 or chunk.dtype != np.float32:
            raise ValueError("Audio must be a 1-D float32 array at 16 kHz")
        if not np.isfinite(chunk).all() or np.max(np.abs(chunk), initial=0) > 1:
            raise ValueError("Audio samples must be finite and between -1 and 1")
        data = np.concatenate((self._pending, chunk))
        complete = len(data) // FRAME_SIZE * FRAME_SIZE
        self._pending = data[complete:].copy()
        matches: list[Match] = []
        for offset in range(0, complete, FRAME_SIZE):
            matches.extend(self._frame(data[offset : offset + FRAME_SIZE]))
        return matches

    def flush(self) -> list[Match]:
        """Process the last partial frame and utterance at end of input."""
        matches: list[Match] = []
        if len(self._pending):
            padded = np.pad(self._pending, (0, FRAME_SIZE - len(self._pending)))
            self._pending = np.empty(0, dtype=np.float32)
            matches.extend(self._frame(padded))
        if self._active:
            matches.extend(self._finish())
        return matches

    def stream(self, chunks: Iterable[np.ndarray]) -> Iterator[Match]:
        """Yield matches from any stream of 16 kHz mono float32 chunks."""
        self.warmup()
        self.reset()
        for chunk in chunks:
            yield from self.feed(chunk)
        yield from self.flush()

    def listen(self, *, device: str | int | None = None) -> Iterator[Match]:
        """Yield matches continuously from a microphone until stopped."""
        from .audio import MicrophoneStream

        self.warmup()
        with MicrophoneStream(device=device) as microphone:
            yield from self.stream(microphone)

    def _frame(self, frame: np.ndarray) -> list[Match]:
        self._last_vad_probability = self.vad.probability(frame)
        spoken = self._last_vad_probability >= self.threshold
        matches: list[Match] = []
        if not self._active:
            if spoken:
                self._start_sample = max(0, self._processed - len(self._preroll) * FRAME_SIZE)
                self._active = [*self._preroll, frame.copy()]
                self._preroll.clear()
                self._silence = 0
                self._stage = "speech"
            else:
                self._preroll.append(frame.copy())
        else:
            self._active.append(frame.copy())
            self._silence = 0 if spoken else self._silence + FRAME_SIZE
            if self._silence >= self.silence_samples or len(self._active) * FRAME_SIZE >= self.max_samples:
                matches.extend(self._finish())
        self._processed += FRAME_SIZE
        return matches

    def _finish(self) -> list[Match]:
        audio = np.concatenate(self._active)
        end = self._start_sample + len(audio)
        self._active = []
        self._silence = 0
        self._utterances += 1
        started = time.perf_counter()
        try:
            self._stage = "transcribing"
            transcript = self.transcriber.transcribe(audio).strip()
            found = []
            for word, pattern in zip(self.wakewords, self._patterns):
                if match := pattern.search(transcript):
                    found.append((match.start(), -len(match.group()), word, match))
            if not found:
                return []
            _, _, word, match = min(found, key=lambda item: (item[0], item[1]))
            end_seconds = end / SAMPLE_RATE
            if end_seconds - self._last_match.get(word, float("-inf")) < self.cooldown_seconds:
                return []
            speaker = None
            similarity = None
            if self.speaker_verifier is not None:
                self._stage = "verifying"
                decision = self.speaker_verifier.verify(audio)
                similarity = decision.similarity
                self._last_speaker_similarity = similarity
                if not decision.accepted:
                    self._speaker_rejections += 1
                    return []
                speaker = decision.speaker
            self._last_match[word] = end_seconds
            command = transcript[match.end() :].lstrip(" \t\r\n,.:;!?-")
            event = Match(
                word, transcript, command, self._start_sample / SAMPLE_RATE, end_seconds,
                speaker=speaker, speaker_similarity=similarity,
            )
            if self.on_wake is not None:
                self._stage = "callback"
                self.on_wake(event)
            if self.router is not None and command:
                self._stage = "running_function"
                event = replace(event, action_result=self.router.run(command))
            self._matches += 1
            return [event]
        finally:
            self._last_inference_ms = (time.perf_counter() - started) * 1000
            self._stage = "idle"
