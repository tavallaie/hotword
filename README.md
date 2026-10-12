# hotword

A streaming Python wakeword library. It captures microphone audio continuously,
uses Silero VAD to find speech, and transcribes each utterance with Whistle. A
matched phrase can call a Python function and pass the remaining words to a
command router. The VAD, transcriber, and router are replaceable objects.

Install the project with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

The detector loads its configured models before microphone capture starts, so
model downloads do not interrupt live audio. On the first run it fetches the
Cactus engine and Whistle weights from Hugging Face. Later runs use the local
cache, including offline. The Silero ONNX model is bundled with this package
and runs through ONNX Runtime. Torch is not required.

```python
import needle
from hotword import NeedleRouter, SileroVAD, WakewordDetector, WhistleTranscriber


@needle.tool
def set_lights(room: str, on: bool) -> str:
    """Switch the lights in a room on or off."""
    return f"{room}: {'on' if on else 'off'}"


detector = WakewordDetector(
    ["hey cactus"],
    vad=SileroVAD(),
    transcriber=WhistleTranscriber(keywords=["hey cactus"]),
    on_wake=lambda match: print("Heard:", match.transcript),
    router=NeedleRouter([set_lights]),
)

for match in detector.listen():
    print(match.command, match.action_result)
```

The CLI also listens continuously on the default microphone:

```sh
uv run hotword --wakeword "hey cactus"
```

Add `--verbose` to see an idle heartbeat every two seconds, including CPU use
(percent of one core), resident memory, queued microphone audio, VAD score,
utterance count, and the last pipeline time. `stage=idle` means the microphone
and VAD are running while Whistle and the command router wait for speech.

Use `--device` to choose another microphone. Stop with Ctrl-C. Microphone
capture runs in a callback and buffers audio while Whistle or Needle works.
If processing falls more than ten seconds behind, listening raises an error
instead of dropping audio without notice.
The CLI checks the microphone before downloading models and reports when the
models are ready. If PortAudio cannot connect to the audio server, it exits
with an error instead of waiting indefinitely.

`NeedleRouter` only runs after the wakeword. It receives the words following
the phrase and executes the registered `@needle.tool` functions. `on_wake`
runs for every accepted match, even if no command follows. An utterance
produces at most one match, and the default one-second cooldown applies per
phrase. A command must be in the same utterance as the wakeword.

To use Whisper, install `uv sync --extra whisper` and pass
`WhisperTranscriber(model="tiny")` as `transcriber`. Any other speech model can
implement `transcribe(samples: np.ndarray) -> str`. A VAD implements
`probability(frame: np.ndarray) -> float` and `reset()`. A router implements
`run(command: str)`. These interfaces live in `hotword.core`; the built-in
adapters live in `hotword.backends`. For an audio stream supplied by another
app, pass an iterable of 16 kHz mono `float32` chunks to `detector.stream(chunks)`.
It yields matches as utterances finish. `feed(samples)` also accepts individual
chunks when another system controls the audio callback. Call `flush()` when
that stream ends and `reset()` before a new stream. Call `detector.warmup()`
before starting an external audio callback so model loading cannot stall it.

Speaker verification is optional. The quantized
[WeSpeaker ONNX model](https://huggingface.co/talatapp/wespeaker-voxceleb-resnet34-LM-onnx)
accepts raw 16 kHz samples, so it uses the ONNX Runtime already installed for
Silero. Its weights are about 7 MB and are not bundled or run during idle
listening. Download `voxceleb_resnet34_LM.int8.onnx` to a local path, then enroll a
speaker from at least three microphone utterances:

```python
import numpy as np
from hotword import (
    EnrolledSpeakerVerifier, MaskedOnnxSpeakerEmbedder, MicrophoneStream,
    NeedleRouter, SileroVAD, WakewordDetector, WhistleTranscriber,
)

embedder = MaskedOnnxSpeakerEmbedder("models/voxceleb_resnet34_LM.int8.onnx")
verifier = EnrolledSpeakerVerifier(embedder, threshold=0.7)

recordings = []
for _ in range(3):
    input("Press Enter, then speak a different phrase for 3 seconds: ")
    with MicrophoneStream() as microphone:
        frames = iter(microphone)
        recordings.append(np.concatenate([next(frames) for _ in range(94)]))

verifier.enroll("owner", recordings)
verifier.save("owner-profile.json")

# In later sessions:
verifier.load("owner-profile.json")
detector = WakewordDetector(
    ["hey cactus"], SileroVAD(), WhistleTranscriber(keywords=["hey cactus"]),
    speaker_verifier=verifier, router=NeedleRouter([set_lights]),
)
```

Use the `set_lights` tool from the first example.
The verifier runs only after a wakeword match, before callbacks or functions.
It rejects a match if the voice is too short or no profile is enrolled. The
threshold is a starting value; compare several owner and other-speaker samples
and tune it for the intended microphone and room. Voice matching alone is not
reliable protection against a recording or voice clone. The profile JSON holds
biometric data, so store it privately. Another embedding model can implement
`embed(samples: np.ndarray) -> np.ndarray` without changing the detector.
The unquantized raw-audio ReDimNet B2 model is also supported by
`RawOnnxSpeakerEmbedder` if a variable-length model is preferred.

Whistle supports [up to 30 seconds per clip](https://cactuscompute.com/blog/whistle),
so the detector caps each utterance below that limit. The bundled
[Silero VAD v6.2.3 ONNX model](https://github.com/snakers4/silero-vad/tree/v6.2.3/src/silero_vad/data)
is covered by [its MIT license](src/hotword/data/SILERO_LICENSE).
