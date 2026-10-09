"""Speech to text: faster-whisper in its own process, fed through queues."""

import multiprocessing
import os
import queue
from collections.abc import Iterator
from pathlib import Path

from ova.transcript import Line
from ova.vad import Segment


def _add_cuda_libs_to_env() -> None:
    """Put the nvidia-* pip wheels on LD_LIBRARY_PATH for processes we spawn.

    The dynamic loader reads the variable when a process starts, so this has
    no effect on the current process.
    """
    try:
        import nvidia
    except ImportError:  # installed without the `cuda` extra
        return
    dirs = sorted(str(p) for p in Path(nvidia.__path__[0]).glob("*/lib"))
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*dirs, current]).rstrip(":")


def _work(jobs, lines, model: str, device: str, language: str | None) -> None:
    from faster_whisper import WhisperModel

    whisper = WhisperModel(model, device=device, compute_type="int8")
    while (job := jobs.get()) is not None:
        channel, start, audio = job
        parts, info = whisper.transcribe(audio, language=language)
        # ponytail: language is detected once, on the first segment; a noisy
        # first segment locks in the wrong one. Pass `language` to be safe.
        language = language or info.language
        for part in parts:
            if text := part.text.strip():
                # Whisper times are relative to the segment it was given.
                t0, t1 = round(start + part.start, 3), round(start + part.end, 3)
                lines.put(Line(channel, t0, t1, text))
    lines.put(None)


class Transcriber:
    """Submit speech segments, then iterate over the transcript lines."""

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "auto",
        language: str | None = None,
    ):
        _add_cuda_libs_to_env()
        context = multiprocessing.get_context("spawn")
        self._jobs, self._lines = context.Queue(), context.Queue()
        self._process = context.Process(
            target=_work,
            args=(self._jobs, self._lines, model, device, language),
            daemon=True,
        )
        self._process.start()

    def submit(self, channel: str, segment: Segment) -> None:
        self._jobs.put((channel, segment.start, segment.audio))

    def close(self) -> None:
        """No more segments; iteration ends once the queued ones are done."""
        self._jobs.put(None)

    def __iter__(self) -> Iterator[Line]:
        while True:
            alive = self._process.is_alive()
            try:
                line = self._lines.get(timeout=0.5)
            except queue.Empty:
                if not alive:
                    # Nobody reads the jobs any more; do not wait to flush them.
                    self._jobs.cancel_join_thread()
                    raise RuntimeError("Whisper worker died") from None
                continue
            if line is None:
                self._process.join()
                return
            yield line
