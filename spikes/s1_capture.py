"""Spike S1: record microphone and system audio (loopback) side by side.

Question: can `soundcard` on PipeWire/PulseAudio capture both streams for
30 minutes without lost audio and without the two streams drifting apart?

Usage:
    uv run python spikes/s1_capture.py --minutes 30

Writes two 16 kHz mono WAV files to recordings/ and prints a report every
minute. "lag" is how far a stream's sample count trails the wall clock since
its first audio block; a value that keeps growing means lost audio.
"""

import argparse
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

RATE = 16_000
BLOCK = RATE // 10  # 100 ms


def to_pcm16(block: np.ndarray) -> bytes:
    """Convert float32 samples in [-1, 1] to little-endian int16 PCM."""
    return (np.clip(block, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def lag_ms(samples: int, elapsed_s: float, rate: int = RATE) -> float:
    """How far the received sample count trails the wall clock, in ms."""
    return (elapsed_s - samples / rate) * 1000


def dbfs(block: np.ndarray) -> float:
    """RMS level in dBFS; -inf for digital silence."""
    rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
    return 20 * np.log10(rms) if rms > 0 else float("-inf")


class Stream(threading.Thread):
    def __init__(self, name: str, source, path: Path, stop: threading.Event):
        super().__init__(name=name, daemon=True)
        self.source, self.path, self.stop = source, path, stop
        self.samples = 0
        self.peak_dbfs = float("-inf")
        self.t0: float | None = None
        self.error: BaseException | None = None

    def run(self):
        try:
            with (
                wave.open(str(self.path), "wb") as wav,
                self.source.recorder(samplerate=RATE, channels=1) as rec,
            ):
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(RATE)
                while not self.stop.is_set():
                    block = rec.record(numframes=BLOCK)[:, 0]
                    if self.t0 is None:  # clock starts at first audio, not at open
                        self.t0 = time.monotonic() - BLOCK / RATE
                    wav.writeframes(to_pcm16(block))
                    self.samples += len(block)
                    self.peak_dbfs = max(self.peak_dbfs, dbfs(block))
        except BaseException as e:  # reported by the main thread
            self.error = e
            self.stop.set()


def report(streams: list[Stream]) -> None:
    now = time.monotonic()
    parts = []
    for s in streams:
        if s.t0 is None:
            parts.append(f"{s.name}: starting")
            continue
        parts.append(
            f"{s.name}: {s.samples / RATE:7.1f}s "
            f"lag {lag_ms(s.samples, now - s.t0):6.0f} ms "
            f"peak {s.peak_dbfs:6.1f} dBFS"
        )
        s.peak_dbfs = float("-inf")
    a, b = streams
    skew = (a.samples - b.samples) / RATE * 1000
    print(" | ".join(parts) + f" | skew {skew:6.0f} ms", flush=True)


def main() -> None:
    import soundcard as sc  # imported here so tests run without an audio server

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--report-every", type=float, default=60, help="seconds")
    parser.add_argument("--out", type=Path, default=Path("recordings"))
    args = parser.parse_args()

    mic = sc.default_microphone()
    speaker = sc.default_speaker()
    loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
    print(f"mic:      {mic.name}\nloopback: {loopback.name}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stop = threading.Event()
    streams = [
        Stream("mic", mic, args.out / f"s1-{stamp}-mic.wav", stop),
        Stream("loopback", loopback, args.out / f"s1-{stamp}-loopback.wav", stop),
    ]
    for s in streams:
        s.start()

    deadline = time.monotonic() + args.minutes * 60
    try:
        while not stop.wait(args.report_every) and time.monotonic() < deadline:
            report(streams)
    except KeyboardInterrupt:
        pass
    stop.set()
    for s in streams:
        s.join(timeout=2)
        if s.error:
            raise SystemExit(f"{s.name} failed: {s.error!r}")
    report(streams)
    print(f"saved: {streams[0].path} {streams[1].path}")


if __name__ == "__main__":
    main()
