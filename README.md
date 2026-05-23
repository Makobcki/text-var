# text-var

Structured implementation of a multiscale VAR text generation stack with OpenAI-compatible API support.

## Repository layout

- `src/api` — API server and launch integration.
- `src/core` — end-to-end inference pipeline.
- `src/var` — VAR model, loss, checkpointing, and training.
- `src/vqvae` — VQ-VAE model and training.
- `src/data` — token cache and dataset utilities.
- `scripts/` — thin wrappers for CLI entrypoints.
- `tests/` — automated tests.
- `configs/` — JSON configs.
- `datasets/` / `checkpoints/` — default runtime data/model directories.

## Quick start

```bash
pip install -e .
python scripts/train_var.py --config configs/train.json
python scripts/api.py --vqvae-path checkpoints/vqvae.pt --var-path checkpoints/var.pt --tokenizer datasets/tokenizer.json
```
