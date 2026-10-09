"""Command line: `ova record`."""

import argparse
from datetime import datetime
from pathlib import Path

from ova.audio import wav_blocks
from ova.stt import Transcriber
from ova.transcript import append_jsonl, markdown
from ova.vad import Segmenter, SileroVad


def record(args: argparse.Namespace) -> None:
    out = args.out or Path("sessions") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    transcriber = Transcriber(args.model, args.device, args.language)
    segmenter = Segmenter(SileroVad())
    # ponytail: a WAV file is one channel, labelled "others"; the live source
    # adds the microphone as "me".
    for block in wav_blocks(args.from_wav):
        for segment in segmenter.feed(block):
            transcriber.submit("others", segment)
    if segment := segmenter.flush():
        transcriber.submit("others", segment)
    transcriber.close()

    lines = []
    for line in transcriber:
        append_jsonl(out / "transcript.jsonl", line)
        lines.append(line)
    (out / "transcript.md").write_text(markdown(lines) + "\n", encoding="utf-8")
    print(out / "transcript.md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ova")
    commands = parser.add_subparsers(required=True)
    rec = commands.add_parser("record", help="transcribe a meeting")
    rec.add_argument("--from-wav", type=Path, required=True, help="16 kHz mono WAV")
    rec.add_argument("--out", type=Path, help="default: sessions/<timestamp>")
    rec.add_argument("--model", default="large-v3-turbo", help="Whisper model")
    rec.add_argument("--device", default="auto", help="auto, cuda or cpu")
    rec.add_argument("--language", help="e.g. pl; default: detect once")
    rec.set_defaults(run=record)
    args = parser.parse_args(argv)
    args.run(args)
