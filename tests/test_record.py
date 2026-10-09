import json
from pathlib import Path

from ova.cli import main

JFK = Path(__file__).parent / "data" / "jfk.wav"  # public domain, 11 s


def test_record_from_wav_writes_transcript(tmp_path):
    main(
        ["record", "--from-wav", str(JFK), "--out", str(tmp_path)]
        + ["--model", "tiny", "--device", "cpu", "--language", "en"]
    )
    rows = [
        json.loads(row)
        for row in (tmp_path / "transcript.jsonl").read_text().splitlines()
    ]
    assert rows
    assert all(row["channel"] == "others" for row in rows)
    assert 0 <= rows[0]["t0"] < rows[-1]["t1"] <= 11.5
    assert "country" in " ".join(row["text"] for row in rows).lower()
    assert "Inni:" in (tmp_path / "transcript.md").read_text()
