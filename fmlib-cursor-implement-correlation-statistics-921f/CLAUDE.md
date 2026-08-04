# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`sber-amazme-fmlib` is a Python library for training, fine-tuning, and deploying Foundation Model (FM) models on the AmazMe platform (Sberbank). It targets the "body + head" architecture for sequential banking event data.

Python ≥ 3.11. Package manager: **Poetry**. All dependencies are resolved from the internal `sber-osc` PyPI mirror.

## Commands

```bash
# Install dependencies
poetry install

# Run all tests
pytest .

# Run tests with thread optimization (recommended)
OMP_NUM_THREADS=16 pytest .

# Run a single test file
pytest fmlib/data/io/tests/test_parquet_dataset.py

# Lint
ruff check .

# Format
ruff format .

# Run pre-commit hooks manually
pre-commit run --all-files

# Launch training (Hydra-based, requires a YAML config)
python fmlib/training/train.py --config-path path/to/configs --config-name config_name

# Launch inference (see examples/*)
python fmlib/training/evaluate.py --config-path path/to/configs --config-name config_name
```

## Architecture

### Core abstractions

- **`Batch` / `GeneralBatch`** (`fmlib/constants/batches.py`) — `Dict[str, Tensor]` aliases that flow through the entire system. Everything passing between transforms, models, and losses uses these types.

- **`TrainingPipeline`** (`fmlib/pipeline/training_pipeline.py`) — `torch.nn.Module` that owns `model`, `losses`, `eval_transform`, and `training_transform`. Selects the correct transform based on `self.training` flag. The `forward` returns `(transformed_batch, model_outputs, loss)`.

- **`InferencePipeline`** (`fmlib/pipeline/inference_pipeline.py`) — Applies one or more transforms then the model; returns `(list[transformed_batch], list[model_output])`.

### Data layer (`fmlib/data/`)

- **`ParquetDataset`** (`fmlib/data/io/parquet_dataset.py`) — Primary data loader. Reads partitioned Parquet files via PyArrow, supports distributed training via `ReplicasInfoProtocol`. Heavy PySpark preprocessing is expected to happen before training; the library only handles in-batch operations (masking, padding, etc.).
- Column types are defined via the protocol in `fmlib/data/io/implementation/column_protocol.py` — `FlatColumn`, `SequenceColumn`, `NamedColumns`.
- `Metadata` describes schema at load time.

### Transforms (`fmlib/nn/transforms/`)

Transforms are `torch.nn.Module`s that take and return `GeneralBatch`. They are composable. Model-specific transforms live at the top level (`sequence_representation.py`, `feature_transformer.py`, etc.); reusable atomic transforms are in `modular/` (padding, masking, cutting sequences, renaming fields, casting, etc.).

### Models (`fmlib/nn/models/`)

All models follow body/head separation:

| Model | Class | Purpose |
|-------|-------|---------|
| FM CDS sequential (body) | `SequenceRepresentationModel` | Generates user embeddings from event sequences via Transformer |
| FM CDS tabular (head) | `FeatureTransformer` / `UpliftFeatureTransformer` | Classification / uplift on top of embeddings |
| NBA DLT | `DLTClassification` | Ромашка task (current) |
| NBA Trivan | `Trivan` | Ромашка task (legacy) |

`SequenceRepresentationModel` forward pass: `embedding → feature_encoder_block → time embedding concat → transformer → (optional aggregation)`. Returns `{output_name: tensor, "last_hidden_state": tensor}`.

### Neural network building blocks (`fmlib/nn/blocks/`)

`BaseEncoderBlock`, `attention.py`, `encoder.py`, `decoder.py`, `cross_encoder.py`, `ffn.py`. Rotary embeddings in `blocks/utils/rotary_embeddings.py`.

### Training (`fmlib/training/`)

Entry point is `fmlib/training/train.py` — uses **Hydra** for config and **Accelerate** for distributed execution. The flow: load config → `instantiate_state` → `training_loop`. Callbacks are split into epoch-level (`callbacks/epoch/`) and step-level (`callbacks/step/`). Checkpointing via `SaveCheckpointCallback`.

### ONNX export (`fmlib/onnx/`)

Models expose `get_input_names()`, `get_output_names()`, and `get_dynamic_shapes()` for ONNX conversion via `fmlib/onnx/compiling.py`.

### Configuration

All experiments are configured via YAML using Hydra. Example configs are in `examples/configs/`. The minimum changes to an existing config for a new run are: `training_data.source`, `validation_data.source`, and the checkpoint save path.

## Code Style

- Line length: **128**. Quotes: **double**. Enforced by `ruff` (see `pyproject.toml` for full rule set).
- Tests live inside each module under a `tests/` subdirectory (co-located with source).
- Branches: `feature/`, `fix/`, `hotfix/`, `release/` prefixes.
- Versioning: SemVer. Releases published to internal Nexus via Jenkins.
- Notebook outputs must be stripped before committing (pre-commit hook handles this).
