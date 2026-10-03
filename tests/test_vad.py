import wave

import numpy as np
import pytest

from toro.audio import RATE, wav_blocks
from toro.vad import FRAME, Segmenter, SileroVad

SPEECH = np.ones(FRAME, dtype=np.float32)
SILENCE = np.zeros(FRAME, dtype=np.float32)


def is_speech(frame: np.ndarray) -> float:
    return float(frame[0])


def stream(*parts: tuple[np.ndarray, int]) -> np.ndarray:
    return np.concatenate([np.tile(frame, count) for frame, count in parts])


def test_segment_ends_after_silence_and_keeps_padding():
    seg = Segmenter(is_speech)
    out = seg.feed(stream((SILENCE, 20), (SPEECH, 30), (SILENCE, 20)))
    assert len(out) == 1
    # 6 frames (~0.2 s) of padding on each side of the 30 speech frames
    assert out[0].start == pytest.approx(14 * FRAME / RATE)
    assert len(out[0].audio) == (6 + 30 + 6) * FRAME
    assert seg.flush() is None


def test_block_size_does_not_change_the_result():
    audio = stream((SILENCE, 20), (SPEECH, 30), (SILENCE, 20))
    seg = Segmenter(is_speech)
    out = [s for i in range(0, len(audio), 1600) for s in seg.feed(audio[i : i + 1600])]
    assert len(out) == 1
    assert len(out[0].audio) == 42 * FRAME


def test_short_click_is_dropped():
    seg = Segmenter(is_speech)
    assert seg.feed(stream((SPEECH, 2), (SILENCE, 20))) == []


def test_long_speech_is_cut_at_max_length():
    seg = Segmenter(is_speech, max_segment_s=1.0)  # 31 frames
    out = seg.feed(stream((SPEECH, 70)))
    assert [len(s.audio) // FRAME for s in out] == [31, 31]
    assert out[1].start == pytest.approx(out[0].end)
    assert len(seg.flush().audio) == 8 * FRAME


def test_flush_returns_segment_in_progress():
    seg = Segmenter(is_speech)
    assert seg.feed(stream((SPEECH, 30))) == []
    assert len(seg.flush().audio) == 30 * FRAME


def test_silero_scores_silence_low():
    vad = SileroVad()
    assert max(vad(SILENCE) for _ in range(10)) < 0.5


def test_wav_blocks_reads_and_validates(tmp_path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.full(4000, 16384, "<i2").tobytes())
    blocks = list(wav_blocks(path))
    assert [len(b) for b in blocks] == [1600, 1600, 800]
    assert blocks[0][0] == pytest.approx(0.5)

    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"\0" * 8)
    with pytest.raises(ValueError):
        list(wav_blocks(path))
