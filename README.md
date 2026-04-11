# NUCLEUS

This repository contains the source code for NUCLEUS, a suite of tools for constructing and evaluating ML surrogates for simulations of two-phase liquid flows. Two-phase heat-transfer is the most efficient known form of cooling, but is still not fully understood and has a large design space. These simulations can be used to model different configurations of liquid cooling. Current work explores creating surrogates that can model different liquids, heater temperatures, and liquid temperatures (for saturated and subcooled nucleate boiling).

## Installation

The core of NUCLEUS is a PyTorch library that can be installed from source using [uv](https://github.com/astral-sh/uv). 
Dependencies are listed in `pyproject.toml`.

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

Run the test suite to verify that the installation was successful:

```bash
uv run pytest test/
```

## Dataset

The datasets used for training are hosted on and available for download from [Huggingface](https://huggingface.co/datasets/hpcforge/BubbleML_2)

## Repository Usage

### Model Imports

The core of nucleus is a library, so the model implementations (and all component modules) can be imported and experimented with in your own code:

```python
from nucleus.models import Nucleus1NeighborMoE
model = Nucleus1NeighborMoE(*args, **kwargs)
```

### Training Scripts

Experiment configurations use [hydra](https://github.com/facebookresearch/hydra) and all config files are stored in `config/`.
Settings can be changed by modifying the config files and can overriden on the command line:

```bash
python scripts/train.py batch_size=8 history_time_window=16
```

### Evaluation Scripts

Inference scripts reuse the config files used for training. The only requirement is to specify 1. the trained
model checkpoint that you want to evaluate and 2. the corresponding `model_cfg` file.

```bash
python scripts/inf.py --checkpoint_path=/path/to/model --model_cfg=config_for_checkpoint
```