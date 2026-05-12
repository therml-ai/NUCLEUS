# NUCLEUS

[![NUCLEUS-Unit-Test-CPU](https://github.com/therml-ai/NUCLEUS/actions/workflows/main.yml/badge.svg)](https://github.com/therml-ai/NUCLEUS/actions/workflows/main.yml)


| ![Subcooled poolboiling of FC-72 with a heater temperature at 97 degrees celsius](media/subcooled_poolboiing_fc72_97c.gif) |
|:-:|
| *Sample trajectory for subcooled poolboiling of FC-72 with a heater temperature of 97 °C* |

This repository contains the source code for NUCLEUS: an ML-based surrogate for simulations of two-phase heat transfer (boiling!). Two-phase heat-transfer is an extremely efficient form of cooling, but it's dynamics are not fully understood and massive a large design space. Typically, two-phase cooling systems are developed with numerical simulations and experimentation. The main issue with these is the time investment. High-fidelity simulations may require compute-days and experiments require manufacturing a cooling system and acquiring or developing different coolant liquids, etc.

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
from nucleus.models import Nucleus1NeighborMoE
model = Nucleus1NeighborMoE(*args, **kwargs)
```

Different models may have different `__init__` paramters, these should match the fields listed in the corresponding `config/model_cfg`.

Each model defines a `forward(x: CollatedBatch)` function. The input is always a `CollatedBatch`, which is defined in `src/nucleus/data/batching.py`

### Training Scripts

Experiment configurations use [hydra](https://github.com/facebookresearch/hydra) and all config files are stored in `config/`.
Settings can be changed by modifying the config files and can overriden on the command line. A good default is

```bash
python scripts/train.py \
    model_cfg=neighbor_moe/neighbor_moe_exp \
    data_cfg=poolboiling \
    normalizer_cfg=standard \
    model_cfg.params.patch_size=16
```

### Evaluation Scripts

The inference scripts share the config files used for training. The only requirement is to specify 1. the trained
model checkpoint that you want to evaluate and 2. the corresponding `model_cfg` file.

```bash
python scripts/inf.py \
    --checkpoint_path=/path/to/trained/model \
    --model_cfg=config_for_checkpoint \
    --model_cfg.params.patch_size=16
```

It is important that the model cfg matches the settings used for training. Including any parameters that were overriden
on the command line when running the training script.