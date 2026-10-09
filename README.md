# O.V.A

**A private AI meeting assistant that runs entirely on your laptop.**

O.V.A (On-Device Voice Assistant) listens to your online meetings (Google Meet, Teams, Zoom, or anything else that plays sound on your computer). It produces a live transcript, tells speakers apart, and writes structured notes with decisions and action items. It does not join the call as a bot, needs no integration with the meeting platform, and never sends audio or text to the cloud.

> **Status:** early development. Working today: `ova record --from-wav` replays a recording through voice activity detection and Whisper, and writes a JSONL and Markdown transcript. Live capture, AI notes, and the desktop window are next. See [Roadmap](#roadmap).

## Benchmarks

Measured on a laptop **RTX 2060 (6 GB VRAM)**, the minimum supported GPU. Scripts: [`spikes/s2_vram.py`](spikes/s2_vram.py) and [`spikes/s3_summary.py`](spikes/s3_summary.py).

### Speech-to-text and LLM on one 6 GB GPU

Whisper `large-v3-turbo` (faster-whisper, `int8_float16`) transcribes while `llama-server` summarises with Qwen3-4B (Q4_K_M, 8k context, `q8_0` KV cache).

| Metric | Result |
|---|---|
| Whisper real-time factor, alone | **0.054** (about 18× faster than real time) |
| Whisper real-time factor, LLM generating at the same time | **0.157** (about 6× faster than real time) |
| LLM generation, Whisper transcribing at the same time | 37–59 tokens/s |
| Peak VRAM, both models plus the desktop | **5098 MiB** of 5729 MiB usable |
| Whisper VRAM | 1080 MiB loaded, 1285 MiB peak |
| LLM VRAM at 8k context | 3220 MiB |
| LLM VRAM at 16k context | 3870 MiB; Whisper then runs out of memory |

Real-time factor is processing time divided by audio length, so lower is faster.

### Bielik vs. Qwen for Polish meeting notes

Same Polish transcript (13,940 characters), same prompt, Q4_K_M, 8k context, each model alone on the GPU.

| | Qwen3-4B-Instruct-2507 | Bielik-4.5B-v3.0-Instruct |
|---|---|---|
| Prompt tokens for the same Polish text | 5030 | **3752** (25% fewer) |
| Prompt processing | **1829 tokens/s** | 1433 tokens/s |
| Generation | 64 tokens/s | 63 tokens/s |
| Peak VRAM, LLM plus the desktop | 3806 MiB | 3796 MiB |
| Polish grammar | Errors | Correct |
| Invented decisions, first prompt → revised prompt | 4 → 5 | 2 → **0** |

Bielik is the default model for the 6 GB profile. Its tokenizer is efficient only for Polish: on an English transcript it needs 31% more tokens than Qwen (4932 vs. 3754).

These are single runs per model on podcast recordings that contain no real decisions. A test on real meetings is still to do.

---

## The problem

Meetings produce decisions, but the decisions get lost.

- **Someone has to take notes.** That person is half-present in the discussion, and the notes depend on what they happened to catch.
- **Cloud meeting assistants solve this, but at a cost.** They join as a visible bot, need admin approval and platform integrations, and upload every word of the conversation to a third-party server. For many companies (legal, finance, healthcare, public sector, anything under NDA) that rules them out.
- **Built-in transcription is locked to one platform.** Each tool works only inside its own ecosystem, and the output is a wall of text, not a list of who agreed to do what.

## The solution

O.V.A is a desktop app that each participant runs on their own machine.

- **No bot, no integration.** It captures your microphone and your computer's audio output, so it works with any meeting app, including phone bridges played through the laptop.
- **Fully local.** Speech recognition, speaker separation, and summarisation all run on the laptop's GPU. Nothing leaves the device, so there is no data processing agreement to sign and no vendor to trust.
- **Notes, not just a transcript.** During the meeting it keeps rolling notes. At the end it produces a summary with decisions, action items (owner, task, deadline), and open questions.
- **Your notes count.** You can type quick keywords during the call; the final summary combines them with the transcript.
- **Built for Polish and English.**

## Who it is for

- Professionals in many meetings per day who want to stay in the conversation instead of typing.
- Teams in regulated or confidential environments where cloud transcription is not allowed.
- Freelancers and consultants who need an accurate record of what clients agreed to.
- Anyone who wants meeting notes they own: summaries export to Markdown and drop straight into Obsidian or any notes tool.

## Key features

| Feature | What you get |
|---|---|
| Live transcript | Text appears within a few seconds of someone speaking. |
| "Me" vs. "others" | Your voice and the other side are captured separately, so attribution is reliable. |
| Speaker separation | Remote participants are told apart and can be renamed ("Speaker 2" → "Anna"). Labels are corrected after the meeting. |
| Rolling notes | A running summary updates during the meeting. |
| Final summary | Decisions, action items, and open questions in a consistent structure. |
| Markdown export | One file per meeting, ready for your notes app. |
| Fact check *(planned)* | Flags statements that contradict documents you've loaded, such as a contract or specification, and shows the source passage. |

## Privacy and consent

All processing happens on your device. Audio, transcripts, and notes are stored locally in your user data directory and are never uploaded.

Recording other people is subject to law (in the EU, GDPR). O.V.A shows a permanent recording indicator, and we recommend telling participants that you are taking AI-assisted notes. The assistant is designed to be discreet, not secret.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Linux with PipeWire or PulseAudio | Ubuntu 22.04+ / Fedora |
| GPU | NVIDIA, 6 GB VRAM (e.g. RTX 2060 / 3050 / 4050) | NVIDIA, 8 GB VRAM (e.g. RTX 4060) |
| RAM | 16 GB | 16 GB+ |
| Audio | Headphones (prevents the mic from picking up the speakers) | |

Machines without a GPU can run transcription on CPU at reduced quality, without live notes. Windows and macOS support is planned for a later phase.

## Roadmap

1. **Audio and live transcript.** Capture mic and system audio, detect speech, transcribe on the GPU, and show the transcript in a desktop window.
2. **AI notes.** Run a local language model for rolling notes and a structured end-of-meeting summary, including the user's own notes.
3. **Speaker separation.** Tell remote participants apart live, with an exact re-labelling pass after the meeting.
4. **Fact check.** Load your documents and flag statements that contradict them.
5. **One-command install.** Model download on first run and a hardware check (`ova doctor`), so anyone can set it up in under 10 minutes.

## License

[MIT](LICENSE)

---

## Tech stack

All components run locally. The runtime avoids PyTorch where possible to keep the install small. Model choices are provisional and will be confirmed by measurements in phase 1.

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Desktop UI | PySide6 (Qt 6, LGPL) | Tray icon with a regular-window fallback (GNOME needs the AppIndicator extension) |
| Audio capture | `soundcard` on PipeWire / PulseAudio | Mic plus the output monitor source, 16 kHz mono; resampling is done by the sound server |
| Voice activity detection | Silero VAD (ONNX, CPU) | 32 ms frames; segments close on ~500 ms of silence or at ~15 s |
| Speech-to-text | faster-whisper (CTranslate2), `large-v3-turbo`, INT8 | GPU, ~1–1.5 GB VRAM, word-level timestamps |
| Speaker separation | Speaker embeddings (WeSpeaker ResNet34, ONNX, CPU) + online clustering; offline re-clustering after the meeting | Applied only to the system-audio channel |
| LLM serving | `llama.cpp` (`llama-server`), GGUF Q4_K_M, quantised KV cache (`q8_0`) | Subprocess with an OpenAI-compatible API; JSON-schema-constrained output |
| LLM models | ~4B class (Qwen3-4B or Bielik-4.5B-v3) on 6 GB; ~8B class on 8 GB | To be benchmarked on Polish meeting transcripts |
| Retrieval *(planned)* | BGE-M3 (ONNX, CPU) dense + sparse, Qdrant in embedded mode, RRF fusion | No database server |
| Contradiction check *(planned)* | mDeBERTa-v3 XNLI (ONNX Runtime, CPU) | Natural-language inference between a claim and a source passage |
| Concurrency | `multiprocessing` (spawn) + queues; Qt signals to the UI | No external message broker |
| Storage | JSONL transcripts and Markdown summaries in `platformdirs` user data dir | |
| Distribution | `uv tool install` / `pipx`; CUDA libraries from pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) | No system CUDA toolkit required |
