"""Wakeword detection with replaceable speech, VAD, and command models."""

from .backends import NeedleRouter, SileroVAD, WhistleTranscriber, WhisperTranscriber
from .audio import MicrophoneStream
from .core import Match, WakewordDetector
from .speaker import EnrolledSpeakerVerifier, MaskedOnnxSpeakerEmbedder, RawOnnxSpeakerEmbedder, SpeakerDecision

__all__ = [
    "Match",
    "MaskedOnnxSpeakerEmbedder",
    "EnrolledSpeakerVerifier",
    "MicrophoneStream",
    "NeedleRouter",
    "RawOnnxSpeakerEmbedder",
    "SileroVAD",
    "SpeakerDecision",
    "WakewordDetector",
    "WhistleTranscriber",
    "WhisperTranscriber",
]
