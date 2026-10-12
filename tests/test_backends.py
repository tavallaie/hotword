import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from hotword import NeedleRouter, SileroVAD, WhistleTranscriber
from hotword.backends import _prepare_whistle_assets


class BackendTests(unittest.TestCase):
    def test_whistle_assets_download_once_without_config_request(self):
        from needle.agent import fetch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tag = fetch._platform_tag()
            lib = fetch.lib_name(3, tag)
            stem, suffix = lib.rsplit(".", 1)
            wheel = root / "engine.whl"
            with ZipFile(wheel, "w") as archive:
                archive.writestr(f"needle/{stem}3.{suffix}", b"engine")
            source_weights = root / "downloaded.cact"
            source_weights.write_bytes(b"weights")
            downloads = []

            def download(*, filename, **_kwargs):
                downloads.append(filename)
                return str(wheel if filename.endswith(".whl") else source_weights)

            with (
                patch.object(fetch, "cache_dir", side_effect=lambda generation: str(root / str(generation))),
                patch("huggingface_hub.hf_hub_download", side_effect=download),
            ):
                cached = _prepare_whistle_assets()
                self.assertEqual(Path(cached).read_bytes(), b"weights")
                self.assertEqual((root / "3" / fetch.lib_name(3)).read_bytes(), b"engine")
                self.assertEqual(len(downloads), 2)
                self.assertNotIn("config.json", downloads)
                self.assertEqual(_prepare_whistle_assets(), cached)
                self.assertEqual(len(downloads), 2)

    def test_silero_onnx_state_resets_between_streams(self):
        backend = SileroVAD()
        silence = np.zeros(512, dtype=np.float32)
        first = backend.probability(silence)
        second = backend.probability(silence)
        backend.reset()
        after_reset = backend.probability(silence)
        self.assertAlmostEqual(first, 0.0016698, places=5)
        self.assertNotEqual(first, second)
        self.assertAlmostEqual(first, after_reset, places=7)

    def test_whistle_passes_audio_and_options_to_sdk(self):
        calls = []

        class FakeWhistle:
            def __init__(self, *, weights):
                calls.append(("load", weights))

            def transcribe(self, audio, **options):
                calls.append((audio.copy(), options))
                return {"text": "hey cactus"}

        with patch.dict(sys.modules, {"needle": types.SimpleNamespace(Whistle=FakeWhistle)}):
            backend = WhistleTranscriber(keywords=["hey cactus"], language="en", weights="test.cact")
            audio = np.zeros(512, dtype=np.float32)
            self.assertEqual(backend.transcribe(audio), "hey cactus")
        self.assertEqual(calls[0], ("load", "test.cact"))
        self.assertTrue(np.array_equal(calls[1][0], audio))
        self.assertEqual(calls[1][1], {"language": "en", "keywords": ["hey cactus"]})

    def test_needle_resets_between_commands_and_only_registered_tools_run(self):
        calls = []

        class FakeNeedle:
            def __init__(self, *, tools, weights):
                calls.append(("load", tools, weights))

            def reset(self):
                calls.append("reset")

            def run(self, text):
                calls.append(("run", text))
                return {"results": [text]}

            def close(self):
                calls.append("close")

        tool = lambda: None
        with patch.dict(sys.modules, {"needle": types.SimpleNamespace(Needle=FakeNeedle)}):
            router = NeedleRouter([tool])
            self.assertEqual(router.run("lights on"), {"results": ["lights on"]})
            self.assertEqual(router.run("lights off"), {"results": ["lights off"]})
            router.close()
        self.assertEqual(calls[0], ("load", [tool], None))
        self.assertEqual(calls[1:], ["reset", ("run", "lights on"), "reset", ("run", "lights off"), "close"])


if __name__ == "__main__":
    unittest.main()
