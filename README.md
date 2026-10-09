# O.V.A

**A private AI meeting assistant that runs entirely on your laptop.**

O.V.A (On-Device Voice Assistant) listens to your online meetings (Google Meet, Teams, Zoom, or anything else that plays sound on your computer). It produces a live transcript, tells speakers apart, and writes structured notes with decisions and action items. It does not join the call as a bot, needs no integration with the meeting platform, and never sends audio or text to the cloud.

> **Status:** early development. The product plan is complete; implementation starts with the core audio and transcription pipeline. See [Roadmap](#roadmap).

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
