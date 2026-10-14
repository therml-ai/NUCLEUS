<h1 align="center" style="margin:0;">
    NUCLEUS
</h1>

<h3 align="center" style="margin: 0; margin-top: 0;">
    ML Surrogates for pool boiling
</h3>
<p align="center">
  <a href="https://doi.org/10.5281/zenodo.20400967"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20400967.svg" alt="DOI"></a>
  <a href="https://github.com/therml-ai/NUCLEUS/actions/workflows/main.yml"><img src="https://github.com/therml-ai/NUCLEUS/actions/workflows/main.yml/badge.svg" alt="Unit test action"></a>
</p>

<figure>
    <img src="media/subcooled_poolboiing_fc72_97c.gif" alt="Subcooled poolboiling of FC-72 with a heater temperature at 97 degrees celsius">
    <div align="center">
        <figcaption>Sample trajectory for subcooled poolboiling of FC-72 with a heater temperature of 97 °C</figcaption>
    </div>
</figure>
<br>


This repository contains the source code for NUCLEUS: an ML-based surrogate for simulations of two-phase heat transfer (boiling!). Two-phase heat-transfer is an extremely efficient form of cooling, but it's dynamics are not fully understood and has a large design space. Typically, two-phase cooling systems are developed with numerical simulations and experimentation. The main issue with these is the time investment. High-fidelity simulations may require compute-days and experiments require manufacturing a cooling system and acquiring or developing different coolant liquids, etc.

We are exploring surrogates as approximate models of boiling that can be used to more rapidly evaluate cooling efficiency for different configurations of different fluid parameters, heater temperatures, and liquid temperatures. Current work focuses on nucleate pool boiling, with future work planned to expand this scope.

## Installation

The core of NUCLEUS is a PyTorch library that can be installed from source using [uv](https://github.com/astral-sh/uv). 
Dependencies are listed in `pyproject.toml`.

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

Run the unit test suite to verify that the installation was successful:

```bash
uv run pytest test/
```

## BubbleML Dataset

We train and evaluate our models using the BubbleML dataset. This is hosted and available for download from [Huggingface](https://huggingface.co/datasets/hpcforge/BubbleML_2).

## Repository Usage

### Model Imports

The core of NUCLEUS is a library, so the model implementations (and all component modules) can be imported and experimented with in your own code:

```python
from boiling_data.flashx.normalizer import NormalizerWrapper
from nucleus.models import get_model

normalizer = NormalizerWrapper.from_json("statistics.json", vector_fields=[("velx", "vely")])
model = get_model("nucleus2_moe", normalizer, patch_size=16, embed_dim=512, ...)
output = model(normalizer.normalize(batch))  # batch: a boiling-data FlashXBatch
```

Model parameters match the fields listed in the corresponding `config/model_cfg`.
Each model's `forward` takes a normalized `FlashXBatch` holding the model's `field_names`, with a config
tensor of its `config_keys`, and returns a normalized `FlashXBatch` prediction.

### Data

Training reads BubbleML files written by [boiling-data](https://github.com/therml-ai/boiling-data), e.g. with
its `scripts/flashx_to_bubbleml.py`, randomly split into train, val and test cases with its
`scripts/split_dataset.py`, along with normalization statistics computed over the training cases. With
boiling-data's scripts:

```bash
python scripts/split_dataset.py /path/to/data/*/*.hdf5 --output-dir /path/to/split --seed 0
python scripts/bubbleml_statistics.py --split-dir /path/to/split \
    --fields dfun temperature velx vely velfacex velfacey
```

NUCLEUS then takes the split directory, `split_dir`, and reads its cases from the `split.json` it holds.

### Training Scripts

Experiment configurations use [hydra](https://github.com/facebookresearch/hydra) and all config files are stored in `config/`.
Each model has its own training script and config, `scripts/train/train_<model>.py` and `config/train_<model>.yaml`.
Settings can be changed by modifying the config files and can overriden on the command line:

```bash
python scripts/train/train_nucleus2_moe.py \
    split_dir=/path/to/split \
    log_dir=/path/to/logs \
    model_cfg.params.patch_size=16
```

To check the data, model and training loop in a few steps, without wandb or checkpoints, run the smoke test on a
split; any further arguments override the model's training config:

```bash
python scripts/train/smoke_test.py /path/to/split --model nucleus2_moe model_cfg.params.processor_blocks=2
```

Pass `checkpoint_path` to resume a run.

### Evaluation Scripts

A rollout needs only the trained checkpoint, which holds the model's config and normalization statistics. Each test case
of the split is rolled out from its history frames and written as a BubbleML file in a `rollouts` directory next to the
checkpoint, at the case's path within the split. The model's architecture comes from the checkpoint, but how it is
run, such as the divfree model's physics options, comes from `model_cfg`, which must name the checkpoint's model; it
defaults to `nucleus2/nucleus2_divfree`.

```bash
python scripts/rollout.py \
    checkpoint_path=/path/to/logs/run/checkpoints/best.pt \
    split_dir=/path/to/split \
    model_cfg=nucleus2/nucleus2_experiment \
    trajectory_steps=300
```
