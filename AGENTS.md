
## Project Background

- **Tech Stack**: Python 3.10, PyTorch, Pytorch Lightning, Hydra config manager.
- **File Structure**:
  - `scripts/`: source code with scripts for training and evaluation.
  - `src/nucleus`: source code with model implementations, Lightning Modules, Datasets, and plotting utilities.
  - `config/`: yaml files to configure model training experiments.
  - `test/`: unit tests written with pytest.

## Python Code Style

- Follow PEP 8 and use type hints when available. 
- Always use descriptive variable and function names. Never use single character variable names.
- Prefer small, testable functions (< 30 lines) that have descriptive names.
- Avoid duplicating code. If two functions share a code block, write a separate function implementing the common code.
- Avoid writing comments for things that will be clear from reading the implementation.

## Commands you can use

Setup the project environment for CPU:

```console
uv venv
source .venv/bin/activate
uv sync --extra cpu --no-cache
```

Run the unit tests:

```console
python -m pytest test/
```

All unit tests should pass. Some may be skipped. No tests should fail.
You should not edit a correct, but failing test to force it to pass. You should
always correct the code in `src/nucleus`.