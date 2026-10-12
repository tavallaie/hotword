"""Optional speaker verification using small raw-audio ONNX models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class SpeakerEmbedder(Protocol):
    def embed(self, samples: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class SpeakerDecision:
    accepted: bool
    speaker: str | None
    similarity: float | None


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    magnitude = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or not np.isfinite(magnitude) or magnitude == 0:
        raise ValueError("Speaker model returned an invalid embedding")
    return value / magnitude


class RawOnnxSpeakerEmbedder:
    """Run a model that accepts [batch, samples] float32 audio at 16 kHz.

    ReDimNet B2 is one compatible model. No extra inference package is needed:
    ONNX Runtime is already used by Silero VAD.
    """

    def __init__(self, model_path: str | Path, *, threads: int = 1) -> None:
        self.model_path = Path(model_path)
        self.threads = threads
        self._session = None

    def warmup(self) -> None:
        if self._session is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Speaker model not found: {self.model_path}")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        session = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1:
            raise ValueError("Speaker model must have one raw-audio input")
        self._session = session

    def embed(self, samples: np.ndarray) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or len(audio) < 16000 or not np.isfinite(audio).all():
            raise ValueError("Speaker verification needs at least one second of 16 kHz mono audio")
        self.warmup()
        assert self._session is not None
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: audio[None, :]})[0]
        if output.shape[0] != 1:
            raise ValueError("Speaker model returned an unexpected batch size")
        return _unit(output[0])


class MaskedOnnxSpeakerEmbedder:
    """Run the quantized WeSpeaker model with [3, 160000] audio and [3, 589] mask."""

    def __init__(self, model_path: str | Path, *, threads: int = 1) -> None:
        self.model_path = Path(model_path)
        self.threads = threads
        self._session = None

    def warmup(self) -> None:
        if self._session is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Speaker model not found: {self.model_path}")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        session = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])
        if {item.name for item in session.get_inputs()} != {"waveform", "mask"}:
            raise ValueError("Expected a speaker model with waveform and mask inputs")
        self._session = session

    def embed(self, samples: np.ndarray) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or len(audio) < 16000 or not np.isfinite(audio).all():
            raise ValueError("Speaker verification needs at least one second of 16 kHz mono audio")
        self.warmup()
        assert self._session is not None
        length = min(len(audio), 160000)
        waveform = np.zeros((3, 160000), dtype=np.float32)
        waveform[0, :length] = audio[:length]
        mask = np.zeros((3, 589), dtype=np.float32)
        mask[0, :max(1, round(length / 160000 * 589))] = 1.0
        output = self._session.run(["embedding"], {"waveform": waveform, "mask": mask})[0]
        if output.shape != (3, 256):
            raise ValueError("Speaker model returned unexpected embedding shape")
        return _unit(output[0])


class EnrolledSpeakerVerifier:
    """Compare utterances with enrolled speakers by cosine similarity."""

    def __init__(self, embedder: SpeakerEmbedder, *, threshold: float) -> None:
        if not -1 <= threshold <= 1:
            raise ValueError("Speaker threshold must be between -1 and 1")
        self.embedder = embedder
        self.threshold = threshold
        self.profiles: dict[str, np.ndarray] = {}

    def warmup(self) -> None:
        prepare = getattr(self.embedder, "warmup", None)
        if prepare is not None:
            prepare()

    def enroll(self, speaker: str, recordings: Sequence[np.ndarray]) -> None:
        if not speaker.strip() or len(recordings) < 3:
            raise ValueError("Provide a speaker name and at least three utterances")
        vectors = [_unit(self.embedder.embed(audio)) for audio in recordings]
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Enrollment embeddings have different dimensions")
        self.profiles[speaker] = _unit(np.mean(vectors, axis=0))

    def verify(self, samples: np.ndarray) -> SpeakerDecision:
        if not self.profiles:
            return SpeakerDecision(False, None, None)
        try:
            vector = _unit(self.embedder.embed(samples))
        except ValueError:
            return SpeakerDecision(False, None, None)
        scores = {
            speaker: float(np.dot(profile, vector))
            for speaker, profile in self.profiles.items()
            if len(profile) == len(vector)
        }
        if not scores:
            return SpeakerDecision(False, None, None)
        speaker = max(scores, key=scores.get)
        score = scores[speaker]
        return SpeakerDecision(score >= self.threshold, speaker if score >= self.threshold else None, score)

    def save(self, path: str | Path) -> None:
        """Save embeddings for later sessions; treat the file as biometric data."""
        data = {"threshold": self.threshold, "profiles": {k: v.tolist() for k, v in self.profiles.items()}}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.profiles = {name: _unit(np.asarray(vector, dtype=np.float32)) for name, vector in data["profiles"].items()}
