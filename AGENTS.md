
## Project Background

- **Tech Stack**: Python 3.10, PyTorch, Hydra config manager, and `boiling-data` for
  data loading (`FlashXBatch`, `FlashXForecastDataset`, `NormalizerWrapper`, augmentations).
- **File Structure**:
  - `scripts/`: `train/train_<model>.py`, one training script per model; `rollout.py`; and analysis scripts.
  - `src/nucleus/models/`: model implementations. A model's `forward` takes a normalized `FlashXBatch`;
    it owns its `NormalizerWrapper` and declares its `field_names` and `config_keys`.
  - `src/nucleus/training/`: the training loop, optimizers, checkpoints and data loaders shared by
    every model, and `modules/`, one training module per model holding its loss and augmentations.
  - `src/nucleus/rollout.py`: autoregressive rollouts, written as BubbleML files.
  - `src/nucleus/physics/` source code for computing physical quantities and enforcing physics.
  - `config/`: yaml files to configure model training experiments, one `train_<model>.yaml` per model.
  - `test/`: unit tests written with pytest. All unit tests should go here.

dependencies are managed using `uv`.

## Python Code Style

- Follow PEP 8 and use type hints when available. 
- Always use descriptive variable and function names. Never use single character variable names.
- Prefer small, testable functions (< 30 lines) that have descriptive names.
- Avoid duplicating code. If two functions share a large code block, write a separate function implementing the common code.
- Avoid writing comments for things that will be clear from reading the implementation.
- inline comments should explain WHY a particular coding approach is used, not WHAT is being done
- *DO NOT* add large docstrings at the top of files or functions.

## Commands you can use

Setup the project environment for CPU:

```console
uv venv
source .venv/bin/activate
uv sync --extra cpu --no-cache
```

Similarly, for GPU

```console
uv venv
source .venv/bin/activate
uv sync --extra cu130 --no-cache
```

Run the unit tests:

```console
python -m pytest test/
```

All unit tests should pass or be skipped. No tests should fail.
You should not edit a correct, but failing test to force it to pass. You should
always correct the code in `src/nucleus`.

Any new unit tests should be written in a subdirectory of `test/`. You should not
write tests in the `src/` or `scripts/` directories.

Check a training script end to end in a few steps, without wandb or checkpoints:

```console
python scripts/train/smoke_test.py /path/to/split --model nucleus2_moe
```

## Slurm

If a GPU is available, you can run the benchmark

```console
python bench/nucleus2_moe.py \
  --backward # time both the forward and backward pass
```
