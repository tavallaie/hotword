import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from hotword import MicrophoneStream
from hotword.audio import _probe_microphone


class MicrophoneTests(unittest.TestCase):
    def test_callback_buffers_audio_before_consumer_reads(self):
        class FakeInputStream:
            def __init__(self, *, callback, **_options):
                self.callback = callback
                self.active = False

            def start(self):
                self.active = True
                for value in (0.25, 0.5):
                    chunk = np.full((512, 1), value, dtype=np.float32)
                    self.callback(chunk, 512, None, types.SimpleNamespace(input_overflow=False))

            def stop(self):
                self.active = False

            def close(self):
                pass

        fake_module = types.SimpleNamespace(InputStream=FakeInputStream)
        with patch("hotword.audio._probe_microphone"), patch.dict(sys.modules, {"sounddevice": fake_module}):
            with MicrophoneStream(buffer_seconds=0.1) as microphone:
                chunks = iter(microphone)
                self.assertEqual(float(next(chunks)[0]), 0.25)
                self.assertEqual(float(next(chunks)[0]), 0.5)

    def test_microphone_probe_times_out(self):
        from subprocess import TimeoutExpired

        with patch("hotword.audio.subprocess.run", side_effect=TimeoutExpired("python", 6)):
            with self.assertRaisesRegex(RuntimeError, "initialization timed out"):
                _probe_microphone(None)


if __name__ == "__main__":
    unittest.main()
