# text-var

Standalone experimental implementation of a **multiscale VAR (Visual/Vector Autoregressive-style) text token generator** with:

- hierarchical token levels,
- optional local attention for prefix encoding,
- hybrid AR + block-parallel decoding,
- thermodynamic sampling with rollback fallback,
- checkpointing with optimizer/AMP scaler/RNG state restore,
- optional validation and TensorBoard logging,
- SemanticTextVQVAE pretraining utility.

## Features

- **Model core**: `VARTransformer` with multiscale heads and optional early-exit heads.
- **Training**: `train.py` supports AMP, grad accumulation, phase freezing, checkpoint save/resume.
- **Validation**: periodic val loss + perplexity.
- **Sampling**: `sample.py` uses `hybrid_cascade_decode`.
- **VQ-VAE pretraining**: `train_vqvae.py`.

## Installation

### Option A: pip + requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Option B: pyproject (editable)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional accelerators:

```bash
pip install -e '.[accelerate]'
```

## Configuration

- Train config: `config/train.json`
- Sample config: `config/sample.json`

Important train keys:

- `checkpoint_path`
- `resume_from` (optional)
- `validation_every`, `validation_batches`
- `tensorboard_enabled`, `log_dir`
- `optimizer` (`adamw` or `adamw8bit`)
- `amp_enabled`, `amp_dtype`

## Usage

### 1) Train VAR

```bash
python train.py --config config/train.json
```

### 2) Resume training

Set `resume_from` in `config/train.json`, then run:

```bash
python train.py --config config/train.json
```

### 3) Run sampling

```bash
python sample.py --config config/sample.json
```

### 4) Pretrain SemanticTextVQVAE

```bash
python train_vqvae.py --output checkpoints/vqvae.pt --steps 500 --batch-size 8 --seq-len 128
```

## Checkpoint format

`checkpoint.py` saves:

- model weights,
- optimizer state (if provided),
- AMP `GradScaler` state (if provided),
- RNG states (`torch`, `python`, and CUDA RNG if available),
- step/loss metadata.

This allows reliable resume with reduced divergence after restart.

## Project layout

- `model.py` — VAR model and SDPA decoder blocks
- `loss.py` — multiscale training loss + corruption masking
- `generator.py` — hybrid decoding and thermodynamic sampling
- `train.py` — training loop + validation + TensorBoard
- `sample.py` — inference/sampling entrypoint
- `checkpoint.py` — save/load/restore training state
- `vqvae.py` / `train_vqvae.py` — SemanticTextVQVAE + trainer

## Notes

This repository is experimental and focuses on training/inference mechanics. For production workloads, pin exact CUDA/PyTorch/flash-attn versions per hardware target.
