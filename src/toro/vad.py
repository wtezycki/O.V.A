"""Voice activity detection: cut a 16 kHz mono stream into speech segments."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from toro.audio import RATE

FRAME = 512  # 32 ms, the frame size Silero VAD expects at 16 kHz
CONTEXT = 64  # samples of the previous frame the model sees again


@dataclass
class Segment:
    start: float  # seconds from the start of the stream
    audio: np.ndarray

    @property
    def end(self) -> float:
        return self.start + len(self.audio) / RATE


class SileroVad:
    """Speech probability for consecutive frames of one stream (keeps state)."""

    def __init__(self):
        from faster_whisper.vad import get_vad_model  # ships the ONNX model

        self._session = get_vad_model().session
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        frame = frame.astype(np.float32, copy=False)
        prob, self._h, self._c = self._session.run(
            None,
            {
                "input": np.concatenate([self._context, frame])[None],
                "h": self._h,
                "c": self._c,
            },
        )
        self._context = frame[-CONTEXT:]
        return float(prob.reshape(-1)[0])


class Segmenter:
    """Feed audio blocks, get back finished speech segments.

    A segment ends after `min_silence_s` of silence or at `max_segment_s`.
    """

    def __init__(
        self,
        speech_prob: Callable[[np.ndarray], float],
        threshold: float = 0.5,
        min_silence_s: float = 0.5,
        max_segment_s: float = 15.0,
        pad_s: float = 0.2,
        min_speech_s: float = 0.25,
    ):
        def frames(seconds: float) -> int:
            return max(1, round(seconds * RATE / FRAME))

        self._prob, self._threshold = speech_prob, threshold
        self._min_silence, self._max = frames(min_silence_s), frames(max_segment_s)
        self._pad, self._min_speech = frames(pad_s), frames(min_speech_s)
        self._rest = np.zeros(0, dtype=np.float32)
        self._pre: deque[np.ndarray] = deque(maxlen=self._pad)
        self._frames: list[np.ndarray] = []
        self._index = self._first = self._speech = self._silence = 0

    def feed(self, samples: np.ndarray) -> list[Segment]:
        data = np.concatenate([self._rest, samples])
        whole = len(data) // FRAME * FRAME
        self._rest = data[whole:]
        steps = (self._step(data[i : i + FRAME]) for i in range(0, whole, FRAME))
        return [segment for segment in steps if segment]

    def flush(self) -> Segment | None:
        """Close the segment in progress at the end of the stream."""
        return self._cut() if self._frames else None

    def _step(self, frame: np.ndarray) -> Segment | None:
        speech = self._prob(frame) >= self._threshold
        index, self._index = self._index, self._index + 1
        if not self._frames:
            if not speech:
                self._pre.append(frame)
                return None
            self._first = index - len(self._pre)
            self._frames = [*self._pre, frame]
            self._pre.clear()
            self._speech, self._silence = 1, 0
            return None
        self._frames.append(frame)
        self._speech += speech
        self._silence = 0 if speech else self._silence + 1
        # ponytail: hard cut at max length can split a word; cut at the
        # lowest-probability frame near the limit if WER shows it matters.
        if self._silence >= self._min_silence or len(self._frames) >= self._max:
            return self._cut()
        return None

    def _cut(self) -> Segment | None:
        keep = len(self._frames) - max(0, self._silence - self._pad)
        segment = Segment(
            self._first * FRAME / RATE, np.concatenate(self._frames[:keep])
        )
        enough_speech = self._speech >= self._min_speech
        self._frames, self._silence = [], 0
        return segment if enough_speech else None
