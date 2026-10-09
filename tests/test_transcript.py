import json

from ova.transcript import Line, append_jsonl, markdown


def test_jsonl_gets_one_object_per_line(tmp_path):
    path = tmp_path / "transcript.jsonl"
    append_jsonl(path, Line("me", 0.0, 1.5, "Cześć"))
    append_jsonl(path, Line("others", 2.0, 3.0, "Hej"))
    rows = [json.loads(row) for row in path.read_text().splitlines()]
    assert rows == [
        {"channel": "me", "t0": 0.0, "t1": 1.5, "text": "Cześć"},
        {"channel": "others", "t0": 2.0, "t1": 3.0, "text": "Hej"},
    ]


def test_markdown_labels_channels_and_sorts_by_time():
    text = markdown([Line("others", 3725.0, 3726.0, "Hej"), Line("me", 5.0, 6.0, "Hi")])
    assert text.splitlines() == [
        "**[00:00:05] Ja:** Hi",
        "",
        "**[01:02:05] Inni:** Hej",
    ]
