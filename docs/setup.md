# Setup

This guide covers environment setup and W&B login.

## Requirements

- Python `3.13` (matches `pyproject.toml`).
- GPU optional. CPU works for quick tests, but training is slow.

## Option A: uv (recommended)

Install `uv` if needed: https://docs.astral.sh/uv/

```bash
pip install uv
```

Create the environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies with the right Torch build (pick one):

```bash
uv sync --extra cpu
uv sync --extra cu118
uv sync --extra cu126
uv sync --extra cu128
```

Notes:

- `uv` reads `pyproject.toml` and uses `uv.lock` when available.
- This is the most reproducible path.

## Option B: conda

```bash
conda env create -f environment.yml
conda activate PTI-LDM-VAE
```

Notes:

- The conda file installs the default PyPI Torch build.
- For CUDA, reinstall Torch/TorchVision from the correct CUDA wheel after the env is created.

## W&B (optional)

This project loads environment variables from a `.env` file at the repo root.

Use the provided template:

```
cp .env.example .env
```

Then edit `.env` and fill in your credentials (see comments inside).

Disable logging per run:

```json
"wandb": { "enabled": false }
```
