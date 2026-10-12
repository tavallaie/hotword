import unittest

import numpy as np

from hotword.speaker import EnrolledSpeakerVerifier, MaskedOnnxSpeakerEmbedder


class FakeEmbedder:
    def embed(self, samples):
        return np.asarray(samples, dtype=np.float32)


class SpeakerTests(unittest.TestCase):
    def test_quantized_model_adapter_pads_live_audio_and_masks_silence(self):
        class FakeSession:
            def run(self, outputs, inputs):
                self_ref.assertEqual(outputs, ["embedding"])
                self_ref.assertEqual(inputs["waveform"].shape, (3, 160000))
                self_ref.assertEqual(inputs["mask"].shape, (3, 589))
                self_ref.assertTrue(np.all(inputs["mask"][1:] == 0))
                self_ref.assertTrue(np.all(inputs["mask"][0, 177:] == 0))
                return [np.tile(np.arange(256, dtype=np.float32), (3, 1))]

        self_ref = self
        embedder = MaskedOnnxSpeakerEmbedder("unused.onnx")
        embedder._session = FakeSession()
        vector = embedder.embed(np.ones(48000, dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_enrolled_owner_is_accepted_and_other_speaker_rejected(self):
        verifier = EnrolledSpeakerVerifier(FakeEmbedder(), threshold=0.8)
        verifier.enroll("owner", [np.array([1, 0]), np.array([0.9, 0.1]), np.array([1, 0.1])])
        self.assertEqual(verifier.verify(np.array([1, 0])).speaker, "owner")
        self.assertFalse(verifier.verify(np.array([0, 1])).accepted)


if __name__ == "__main__":
    unittest.main()
