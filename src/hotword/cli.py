"""Listen continuously for a wakeword on the default microphone."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import sys
import time

from .audio import MicrophoneStream, _probe_microphone
from .backends import SileroVAD, WhistleTranscriber, WhisperTranscriber
from .core import WakewordDetector
from .monitor import ResourceMonitor, rss_mib


def main() -> None:
    parser = argparse.ArgumentParser(description="Listen for a spoken wakeword")
    parser.add_argument("--wakeword", action="append", required=True, help="phrase to detect")
    parser.add_argument("--transcriber", choices=("whistle", "whisper"), default="whistle")
    parser.add_argument("--device", help="microphone device name or index")
    parser.add_argument("--verbose", action="store_true", help="report idle CPU, memory, audio queue and detector state")
    args = parser.parse_args()
    transcriber = (
        WhistleTranscriber(keywords=args.wakeword)
        if args.transcriber == "whistle"
        else WhisperTranscriber()
    )
    detector = WakewordDetector(args.wakeword, SileroVAD(), transcriber)
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    try:
        print("Checking microphone...", file=sys.stderr, flush=True)
        _probe_microphone(device)
        print("Loading speech models. First run may download Whistle...", file=sys.stderr, flush=True)
        started = time.monotonic()
        detector.warmup()
        if args.verbose:
            print(f"[hotword] models_ready={time.monotonic() - started:.1f}s rss={rss_mib():.1f}MiB", file=sys.stderr, flush=True)
        print("Speech models ready. Opening microphone...", file=sys.stderr, flush=True)
        with MicrophoneStream(device=device) as microphone:
            print("Listening for " + ", ".join(args.wakeword), file=sys.stderr, flush=True)
            monitor = ResourceMonitor(detector, microphone) if args.verbose else nullcontext()
            with monitor:
                for match in detector.stream(microphone):
                    print(f"{match.wakeword}: {match.command}", flush=True)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        parser.exit(1, f"hotword: {exc}\n")


if __name__ == "__main__":
    main()
