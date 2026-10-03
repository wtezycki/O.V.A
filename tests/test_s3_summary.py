import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "spikes"))  # s3 imports s2_vram

from s3_summary import clip  # noqa: E402


def test_clip_keeps_short_text():
    assert clip("Krótki tekst.", 100) == "Krótki tekst."


def test_clip_ends_on_sentence():
    assert clip("Pierwsze zdanie. Drugie zdanie. Trzecie.", 30) == "Pierwsze zdanie."


def test_clip_without_sentence_end_cuts_hard():
    assert clip("a" * 50, 10) == "a" * 10
