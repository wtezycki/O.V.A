"""Transcript lines: JSONL on disk, Markdown for export."""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

LABELS = {"me": "Ja", "others": "Inni"}  # microphone, loopback


@dataclass
class Line:
    channel: str  # a key of LABELS
    t0: float  # seconds from the start of the recording
    t1: float
    text: str


def append_jsonl(path: Path, line: Line) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(line), ensure_ascii=False) + "\n")


def markdown(lines: Iterable[Line]) -> str:
    def clock(seconds: float) -> str:
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes // 60:02}:{minutes % 60:02}:{secs:02}"

    return "\n\n".join(
        f"**[{clock(line.t0)}] {LABELS[line.channel]}:** {line.text}"
        for line in sorted(lines, key=lambda line: line.t0)
    )
