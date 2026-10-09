# O.V.A

O.V.A (On-Device Voice Assistant): a private, on-device meeting assistant: live transcript, speaker separation, and AI notes on a Linux laptop (NVIDIA 6–8 GB). See `README.md` for the product description.

## Design notes

Plans and decisions live in the Obsidian vault `/home/wiktor/C/Vaults/toro.ai`:
- `ROADMAP.md`: milestones M0–M7 and implementation order (source of truth).
- `ITERATIONS_DESKTOP.md`: technical scope of the desktop variant.

## Commands

```bash
uv sync                    # install dependencies
uv sync --extra cuda       # + CUDA 12 libs from pip (cuBLAS, cuDNN, runtime) for GPU runs
uv run pytest              # tests
uv run ruff check          # lint
uv run ruff format         # format
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, and `pytest` on every PR.

## Layout

- `src/ova/`: application package.
- `tests/`: pytest tests. Tests must run without a GPU or a microphone.

## Rules

- Never commit meeting recordings, transcripts, or model files (see `.gitignore`).
- Trunk-based: small PRs to `main`, Conventional Commits.
