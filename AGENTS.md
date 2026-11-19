
## Project Background

- **Tech Stack**: Python 3.12, PyTorch, Hydra config manager, and `boiling-data` for
  data loading (`FlashXBatch`, `FlashXForecastDataset`, `NormalizerWrapper`, augmentations).
- **File Structure**:
  - `scripts/`: `train/train_<model>.py`, one training script per model; `rollout.py`; and analysis scripts.
  - `src/nucleus/models/`: model implementations. A model's `forward` takes a normalized `FlashXBatch`;
    it owns its `NormalizerWrapper` and declares its `field_names` and `config_keys`. 
  - `src/nucleus/training/`: the training loop, optimizers, checkpoints and data loaders shared by
    every model, and `modules/`, one training module per model holding its loss and augmentations.
  - `src/nucleus/rollout.py`: autoregressive rollouts, written as BubbleML files using the boiling-data repo.
  - `src/nucleus/physics/` source code for computing physical quantities and enforcing physics.
  - `config/`: yaml files to configure model training experiments and rollouts, one `train_<model>.yaml` per model.
  - `test/`: unit tests written with pytest. All unit tests should be in a subdirectory of `test/`.

dependencies are managed using `uv`.

## Python Code Style

- Follow PEP 8 and use type hints when available. 
- Always use descriptive variable and function names. Never use single character variable names.
- Prefer small, testable functions (< 30 lines) that have descriptive names.
- Avoid duplicating code in utilities. If two functions share a large code block, 
  write a separate function implementing the common code. An exception is pytorch model implementations,
  where some duplication is permissible.
- Avoid writing comments for things that will be clear from reading the implementation.
- inline comments should explain WHY a particular coding approach is used, not WHAT is being done
- *DO NOT* add large docstrings at the top of files or functions.
- If code implements an algorithm from a specific paper or book, it should be documented in a comment, but 
  code should stay readable without requiring people to check the reference. I.e., variable names should 
  always be readable and consistent, even if an algorithm is based on a reference.

## Adding a new Model

A new model should implement the following steps:

1. `src/nucleus/models/<model>.py`, registered with `@register_model("<name>")`, and imported in `src/nucleus/models/__init__.py`.
2. `src/nucleus/training/modules/<model>.py`, holding the training module and settings.
3. `config/train_<model>.yaml` and `config/model_cfg/<model_group>/<model>.yaml`.
4. `scripts/train/train_<model>.py`.
5. Add the name to `MODELS` in `scripts/train/smoke_test.py`
6. Add the name to the list in `test/model/test_get_model.py`.
7. Tests in `test/model/` and `test/training/`, including one that composes the train config into the training settings.

## Model Conventions

- nucleus2 models inherit `Nucleus2MoEBase`. Don't add additional base classes for nucleus2.
  A derived nucleus2 model implements:
  - `config_class`, its config dataclass (`Nucleus2MoEBaseConfig` or a subclass of it).
  - `config_from_dict`, a `staticmethod` building `config_class` from a checkpoint's config dict.
  - `field_names`, the fields it reads and predicts.
  - `__init__`, calling `super().__init__(config, normalizer, in_channels)` and setting `self.debed`
    to its output head, e.g. `patch_debed(config, out_channels)`.
  - `forward(batch)`, taking a normalized `FlashXBatch` and returning an output holding the
    `prediction` batch and the `moe_outputs`.
  - Optionally, `vector_fields` and `runtime_options`.
- `step(cells, sim_params)` is tensor-only, because `build_model` compiles it with `fullgraph=True`.
  Physics and anything that works on a `FlashXBatch` belongs in `forward`.
- Normalization: data-only models leave `vector_fields` empty, so each field is normalized on its own.
  Only models whose physics is linear in velocity (divergence, curl, projection) set
  `vector_fields = (("velfacex", "velfacey"),)`, which gives the two components one shared scale and no offset.
- Run Poisson solves in float64. Inputs can be upcast to float64.
- The physics code assumes every sample in a batch has the same grid spacing (`shared_spacing`).
- Options that change how a model runs, rather than what it learned, go in `runtime_options`.
  A checkpoint keeps the options of the model it is loaded into. Runtime options can be toggled
  on or off during inference rollouts. For example, `use_mass_transfer` can be enabled at inference
  to compute the dilational part of the velocity from the temperature field.

## Field Layouts

A 2D simulation is discretized into `H x W` cells.
- Cell-centered fields are `(..., H, W)`
- Some fields are on cell faces:
    - `velfacex` is `(..., H, W+1)`
    - `velfacey` is `(..., H+1, W)`.
- some fields are on cell nodes (corners):
    - For example, stream function of a helmholtz decomposition is `(..., H+1, W+1)`.
- Row index `0` is the bottom of the domain.
- `FlashXBatch.stack_field_cells` gives a face field two channels per cell, the low face then the
  high face (left/right for velfacex, bottom/top for velfacey). The function `unstack_field_cells` averages 
  the two predictions of each interior face.

Variables should be named according to the grid they use (if it's known). For instance, if a velocity
is on a face grid it could be named `velx_faces`. If we have something on a nodal grid, it can be `psi_nodes`.
In general, we do not need to specify that a variable corresponds to a cell-centered grid, unless there are
some conversions between different grid types.

## Project Setup and Tests 

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
