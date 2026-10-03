"""Spike S3: which ~4B model writes better Polish meeting summaries?

Transcribes a Polish recording with Whisper (or reads a ready transcript),
then asks each GGUF model for the same summary through llama-server. Writes
the transcript and one Markdown file with all summaries to recordings/ for
manual review, and prints a JSON report with speed and peak VRAM per model.

Usage (needs `uv sync --extra cuda` and a llama.cpp CUDA build):
    uv run python spikes/s3_summary.py --audio recordings/x-loopback.wav \
        --llama-server ~/.cache/toro/llama.cpp/llama-b11347/llama-server \
        --model qwen.gguf --model bielik.gguf
"""

import argparse
import gc
import json
import urllib.request
from pathlib import Path

from s2_vram import (
    PORT,
    VramSampler,
    ensure_cuda_libs_on_path,
    load_audio,
    start_llama,
    transcribe,
)

SYSTEM = "Jesteś asystentem spotkań. Piszesz po polsku, pełnymi zdaniami."
PROMPT = """Podsumuj poniższy transkrypt rozmowy. Użyj dokładnie tych nagłówków:

## Podsumowanie
2–3 akapity pełnymi zdaniami: o czym była rozmowa i jakie były główne wątki.

## Decyzje
Tylko ustalenia podjęte przez uczestników w trakcie tej rozmowy. Do każdego
dodaj dosłowny cytat z transkryptu w cudzysłowie.

## Action items
Tylko zadania do wykonania po rozmowie: kto, co, do kiedy. Do każdego dodaj
dosłowny cytat z transkryptu w cudzysłowie.

## Otwarte pytania
Tylko pytania, które padły w rozmowie i zostały bez odpowiedzi.

Zasady: opieraj się tylko na transkrypcie. Jeśli sekcja nie ma treści, wpisz
w niej jedno słowo: brak. Nie używaj nawiasów kwadratowych ani wypełniaczy.
Niczego nie zmyślaj.

Transkrypt:
"""


def clip(text: str, max_chars: int) -> str:
    """Cut to max_chars, ending on a full sentence when there is one."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    end = head.rfind(". ")
    return head[: end + 1] if end > 0 else head


def chat(transcript: str, max_tokens: int) -> dict:
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT + transcript},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,  # same for every model, low to limit run-to-run noise
        "seed": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main() -> None:
    ensure_cuda_libs_on_path()

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", type=Path, help="16 kHz mono WAV")
    src.add_argument("--transcript", type=Path, help="text file, skips Whisper")
    p.add_argument("--llama-server", type=Path, required=True)
    p.add_argument("--model", type=Path, action="append", required=True, help="GGUF")
    p.add_argument("--whisper", default="large-v3-turbo")
    p.add_argument("--language", default="pl")
    p.add_argument("--minutes", type=float, default=30, help="audio to transcribe")
    p.add_argument("--ctx", type=int, default=8192, help="S2: 8k fits in 6 GB")
    p.add_argument("--max-chars", type=int, default=14000, help="~5k tokens")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--out", type=Path, default=Path("recordings"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.transcript:
        stem = args.transcript.stem.removeprefix("s3-").removesuffix("-transcript")
        full = args.transcript.read_text()
    else:
        from faster_whisper import WhisperModel

        stem = args.audio.stem
        whisper = WhisperModel(args.whisper, device="cuda", compute_type="int8_float16")
        full, _ = transcribe(
            whisper, load_audio(args.audio, args.minutes * 60), args.language
        )
        del whisper  # free VRAM so each LLM is measured alone
        gc.collect()
        (args.out / f"s3-{stem}-transcript.txt").write_text(full)

    transcript = clip(full, args.max_chars)
    report: dict = {"transcript_chars": len(full), "used_chars": len(transcript)}
    sections = []
    for model in args.model:
        sampler = VramSampler()
        sampler.start()
        llama = start_llama(args.llama_server, model, args.ctx, [])
        try:
            reply = chat(transcript, args.max_tokens)
        finally:
            llama.kill()
            llama.wait()
            sampler.stop.set()
        t = reply["timings"]
        report[model.name] = {
            "prompt_tokens": t["prompt_n"],
            "prompt_per_second": round(t["prompt_per_second"], 1),
            "predicted_tokens": t["predicted_n"],
            "predicted_per_second": round(t["predicted_per_second"], 1),
            "finish_reason": reply["choices"][0]["finish_reason"],
            "peak_vram_mib": sampler.peak,
        }
        text = reply["choices"][0]["message"]["content"]
        sections.append(f"# {model.name}\n\n{text}")

    out = args.out / f"s3-{stem}-summaries.md"
    out.write_text("\n\n---\n\n".join(sections) + "\n")
    print(json.dumps(report, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
