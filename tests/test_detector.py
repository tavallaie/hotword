import unittest
from unittest.mock import patch

import numpy as np

from hotword import SpeakerDecision, WakewordDetector


class FakeVAD:
    def __init__(self):
        self.reset_count = 0

    def probability(self, frame):
        return 0.9 if np.max(np.abs(frame)) > 0.2 else 0.0

    def reset(self):
        self.reset_count += 1


class FakeTranscriber:
    def __init__(self, *texts):
        self.texts = iter(texts)
        self.calls = 0

    def transcribe(self, samples):
        self.calls += 1
        return next(self.texts)


class FakeRouter:
    def __init__(self):
        self.commands = []

    def run(self, command):
        self.commands.append(command)
        return {"ran": command}


class FakeSpeakerVerifier:
    def __init__(self, accepted):
        self.accepted = accepted

    def verify(self, samples):
        return SpeakerDecision(self.accepted, "owner" if self.accepted else None, 0.91 if self.accepted else 0.22)


def utterance():
    return np.concatenate(
        [
            np.zeros(512, dtype=np.float32),
            np.full(512 * 3, 0.5, dtype=np.float32),
            np.zeros(512 * 3, dtype=np.float32),
        ]
    )


class DetectorTests(unittest.TestCase):
    def test_wakeword_runs_callback_and_only_trailing_command(self):
        heard = []
        router = FakeRouter()
        detector = WakewordDetector(
            ["hey cactus"], FakeVAD(), FakeTranscriber("HEY CACTUS, turn on kitchen lights"),
            on_wake=heard.append, router=router, silence_seconds=0.064,
        )
        audio = utterance()
        matches = detector.feed(audio[:1000]) + detector.feed(audio[1000:]) + detector.flush()
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].command, "turn on kitchen lights")
        self.assertEqual(matches[0].action_result, {"ran": "turn on kitchen lights"})
        self.assertEqual(heard[0].wakeword, "hey cactus")
        self.assertEqual(router.commands, ["turn on kitchen lights"])

    def test_no_wakeword_never_runs_function(self):
        router = FakeRouter()
        detector = WakewordDetector(
            ["hey cactus"], FakeVAD(), FakeTranscriber("turn on kitchen lights"),
            router=router, silence_seconds=0.064,
        )
        self.assertEqual(detector.feed(utterance()), [])
        self.assertEqual(detector.flush(), [])
        self.assertEqual(router.commands, [])

    def test_unrecognized_speaker_cannot_run_function_or_callback(self):
        heard = []
        router = FakeRouter()
        detector = WakewordDetector(
            ["hey cactus"], FakeVAD(), FakeTranscriber("hey cactus, turn on lights"),
            on_wake=heard.append, router=router, speaker_verifier=FakeSpeakerVerifier(False),
            silence_seconds=0.064,
        )
        self.assertEqual(detector.feed(utterance()), [])
        self.assertEqual(heard, [])
        self.assertEqual(router.commands, [])
        self.assertEqual(detector.status.speaker_rejections, 1)

    def test_wakeword_without_command_only_calls_callback(self):
        heard = []
        router = FakeRouter()
        detector = WakewordDetector(
            ["hey cactus"], FakeVAD(), FakeTranscriber("Hey cactus!"),
            on_wake=heard.append, router=router, silence_seconds=0.064,
        )
        matches = detector.feed(utterance())
        self.assertEqual(matches[0].command, "")
        self.assertEqual(len(heard), 1)
        self.assertEqual(router.commands, [])

    def test_stream_processes_consecutive_utterances(self):
        detector = WakewordDetector(
            ["hey cactus"], FakeVAD(),
            FakeTranscriber("hey cactus, status", "hey cactus, lights on"),
            silence_seconds=0.064, cooldown_seconds=0,
        )
        audio = np.concatenate([utterance(), utterance()])
        chunks = (audio[i : i + 700] for i in range(0, len(audio), 700))
        matches = list(detector.stream(chunks))
        self.assertEqual([match.command for match in matches], ["status", "lights on"])
        self.assertLessEqual(matches[0].end_seconds, matches[1].start_seconds)

    def test_listen_reads_a_microphone_stream(self):
        seen_devices = []

        class FakeMicrophone:
            def __init__(self, *, device):
                seen_devices.append(device)

            def __enter__(self):
                return iter([utterance()])

            def __exit__(self, *_args):
                pass

        detector = WakewordDetector(
            ["hey cactus"], FakeVAD(), FakeTranscriber("hey cactus, status"),
            silence_seconds=0.064,
        )
        with patch("hotword.audio.MicrophoneStream", FakeMicrophone):
            matches = list(detector.listen(device="USB mic"))
        self.assertEqual(seen_devices, ["USB mic"])
        self.assertEqual(matches[0].command, "status")


if __name__ == "__main__":
    unittest.main()
