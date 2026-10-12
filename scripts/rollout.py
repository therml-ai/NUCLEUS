from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from nucleus.models import load_model
from nucleus.rollout import RolloutSettings, rollout_case, save_rollout
from nucleus.training.data import read_split


@hydra.main(version_base=None, config_path="../config", config_name="rollout")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.checkpoint_path, map_location=device).to(device).eval()
    settings = RolloutSettings(
        start_frame=cfg.start_frame,
        history_timesteps=cfg.history_timesteps,
        trajectory_steps=cfg.trajectory_steps,
        frames_per_step=cfg.frames_per_step,
    )
    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path(cfg.checkpoint_path).parent / "rollouts"
    test_dir = Path(cfg.split_dir) / "test"
    test_cases = read_split(cfg.split_dir)["test"]
    if not test_cases:
        raise SystemExit(f"{cfg.split_dir} has no test cases")
    for path in test_cases:
        prediction, time = rollout_case(model, path, settings, device)
        # the case's path within the split, since cases of different fluids share names
        print(f"wrote {save_rollout(prediction, time, output_dir / path.relative_to(test_dir))}")


if __name__ == "__main__":
    main()
