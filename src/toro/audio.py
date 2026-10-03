"""Audio sources. Every source yields 16 kHz mono float32 blocks."""

import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

RATE = 16_000


def wav_blocks(path: Path, block_s: float = 0.1) -> Iterator[np.ndarray]:
    """Replay a 16 kHz mono 16-bit WAV file in blocks, like a live source."""
    with wave.open(str(path)) as wav:
        fmt = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
        if fmt != (RATE, 1, 2):
            raise ValueError(f"{path}: need 16 kHz mono 16-bit WAV, got {fmt}")
        while frames := wav.readframes(int(block_s * RATE)):
            yield np.frombuffer(frames, "<i2").astype(np.float32) / 32768
