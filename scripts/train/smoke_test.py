"""Runs a few steps of a small version of a model's training on a split directory,
without wandb, checkpoints or DataLoader worker processes, so errors in the data, model
or training loop surface in seconds. Any further arguments override the model's
training config and the small model, e.g. model_cfg.params.processor_blocks=2."""

import argparse
import importlib
from pathlib import Path

from hydra import compose, initialize_config_dir

from nucleus.training.run import run_training

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
MODELS = ("nucleus2_moe", "nucleus2_moe_divfree")
# small enough to run on a CPU; a head of 32 channels is the smallest neighborhood
# attention takes, and the patch size is left to the config, as it must divide the grid
SMALL_MODEL = [
    "model_cfg.params.embed_dim=64",
    "model_cfg.params.num_heads=2",
    "model_cfg.params.processor_blocks=1",
    "model_cfg.params.num_experts=2",
    "model_cfg.params.topk=1",
    "model_cfg.params.moe_intermediate_dim=64",
    "data.batch_size=2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path, help="a directory made by boiling-data's scripts/split_dataset.py")
    parser.add_argument("--model", choices=MODELS, default="nucleus2_moe")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides of config/train_<model>.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # the given overrides come last, so they win over the small model
    overrides = ["smoke=true", f"split_dir={args.split_dir.resolve()}", *SMALL_MODEL, *args.overrides]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=f"train_{args.model}", overrides=overrides)
    # the model's own train script, next to this one, so the smoke test trains with
    # the same training module
    train_script = importlib.import_module(f"train_{args.model}")
    state = run_training(cfg, train_script.build_training_module)
    print(f"smoke test passed: {state.step} training steps of {args.model}")


if __name__ == "__main__":
    main()
