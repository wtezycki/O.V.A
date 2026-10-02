"""Spike S2: do Whisper and a ~4B LLM fit in laptop VRAM at the same time?

Loads faster-whisper large-v3-turbo (INT8) and llama-server with a GGUF
model, then transcribes audio while the LLM summarises a long transcript.
Samples `nvidia-smi` throughout and prints a JSON report: VRAM per phase,
peak VRAM, Whisper real-time factor (solo vs. concurrent), and LLM speed.

Usage (needs `uv sync --extra cuda` and a llama.cpp CUDA build):
    uv run python spikes/s2_vram.py --audio recordings/x.wav \
        --llama-server ~/.cache/toro/llama.cpp/llama-b11347/llama-server \
        --model path/to/model.gguf
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

RATE = 16_000
CHUNK_S = 15  # roughly the longest VAD segment planned for M1
PORT = 8089


def cuda_lib_dirs() -> list[str]:
    """Library dirs of the nvidia-* pip wheels (cuBLAS, cuDNN, CUDA runtime)."""
    import nvidia

    root = Path(nvidia.__path__[0])
    return sorted(str(p) for p in root.glob("*/lib"))


def ensure_cuda_libs_on_path() -> None:
    """Re-exec with LD_LIBRARY_PATH set; the dynamic loader reads it at start."""
    dirs = cuda_lib_dirs()
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d in current.split(":") for d in dirs):
        return
    env = dict(os.environ, LD_LIBRARY_PATH=":".join([*dirs, current]).rstrip(":"))
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def vram_used_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.split()[0])


class VramSampler(threading.Thread):
    def __init__(self, every_s: float = 0.2):
        super().__init__(daemon=True)
        self.every_s, self.peak, self.stop = every_s, 0, threading.Event()

    def run(self):
        while not self.stop.wait(self.every_s):
            self.peak = max(self.peak, vram_used_mib())


def load_audio(path: Path, max_s: float) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2)
        frames = w.readframes(min(w.getnframes(), int(max_s * RATE)))
    return np.frombuffer(frames, "<i2").astype(np.float32) / 32767


def transcribe(model, audio: np.ndarray, language: str | None) -> tuple[str, float]:
    """Transcribe in CHUNK_S pieces like the live pipeline will; return text, RTF."""
    texts, t0 = [], time.monotonic()
    for start in range(0, len(audio), CHUNK_S * RATE):
        segments, _ = model.transcribe(
            audio[start : start + CHUNK_S * RATE], language=language
        )
        texts.extend(s.text for s in segments)
    return " ".join(texts).strip(), (time.monotonic() - t0) / (len(audio) / RATE)


def start_llama(
    server: Path, model: Path, ctx: int, extra: list[str]
) -> subprocess.Popen:
    proc = subprocess.Popen(
        [str(server), "-m", str(model), "-c", str(ctx), "-ngl", "99", "-np", "1",
         "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
         "--host", "127.0.0.1", "--port", str(PORT), *extra],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited with {proc.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
            return proc
        except OSError:
            time.sleep(1)
    proc.kill()
    raise SystemExit("llama-server did not become healthy in 180 s")


def summarise(transcript: str, max_tokens: int) -> dict:
    body = {
        "messages": [
            {"role": "system", "content": "Jesteś asystentem spotkań."},
            {"role": "user", "content": "Podsumuj transkrypt: decyzje, action "
             "items (kto, co, do kiedy), otwarte pytania.\n\n" + transcript},
        ],
        "max_tokens": max_tokens,
        "cache_prompt": False,  # measure full prefill every time
    }  # fmt: skip
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["timings"]


def main() -> None:
    ensure_cuda_libs_on_path()
    from faster_whisper import WhisperModel

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audio", type=Path, required=True, help="16 kHz mono WAV")
    p.add_argument("--llama-server", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="GGUF file")
    p.add_argument("--whisper", default="large-v3-turbo")
    p.add_argument("--compute-type", default="int8_float16")
    p.add_argument("--language", default=None, help="e.g. pl; default: detect")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--prompt-tokens", type=int, default=12000, help="approx.")
    p.add_argument("--solo-minutes", type=float, default=2)
    p.add_argument("--concurrent-minutes", type=float, default=3)
    p.add_argument("--llama-args", default="", help='extra flags, e.g. "-ub 256"')
    args = p.parse_args()

    audio = load_audio(args.audio, (args.solo_minutes + args.concurrent_minutes) * 60)
    split = int(args.solo_minutes * 60 * RATE)
    report: dict = {"config": vars(args) | {"audio": str(args.audio)}}
    report["config"] = {k: str(v) for k, v in report["config"].items()}
    report["baseline_mib"] = vram_used_mib()
    sampler = VramSampler()
    sampler.start()

    whisper = WhisperModel(args.whisper, device="cuda", compute_type=args.compute_type)
    report["after_whisper_load_mib"] = vram_used_mib()
    text, report["whisper_rtf_solo"] = transcribe(whisper, audio[:split], args.language)
    report["peak_whisper_solo_mib"] = sampler.peak

    llama = start_llama(
        args.llama_server, args.model, args.ctx, args.llama_args.split()
    )
    llm_runs: list[dict] = []
    try:
        report["after_llama_load_mib"] = vram_used_mib()
        # Pad the real transcript to roughly a long meeting (~3.5 chars/token).
        long_text = (text + " ") * (
            args.prompt_tokens * 7 // 2 // max(len(text), 1) + 1
        )
        long_text = long_text[: args.prompt_tokens * 7 // 2]

        whisper_done = threading.Event()

        def llm_loop():
            while not whisper_done.is_set():
                llm_runs.append(summarise(long_text, max_tokens=512))

        llm = threading.Thread(target=llm_loop, daemon=True)
        llm.start()
        try:
            _, report["whisper_rtf_concurrent"] = transcribe(
                whisper, audio[split:], args.language
            )
        except RuntimeError as e:  # CUDA OOM is the result we are looking for
            report["error"] = str(e)
        whisper_done.set()
        llm.join(timeout=600)
    finally:
        llama.kill()  # SIGTERM waits for the running request to finish
        llama.wait()
        sampler.stop.set()

    report["peak_total_mib"] = sampler.peak
    report["llm_runs"] = [
        {k: round(r[k], 1) for k in
         ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second")}
        for r in llm_runs
    ]  # fmt: skip
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
