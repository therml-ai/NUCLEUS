import hydra
from omegaconf import DictConfig, OmegaConf

from nucleus.training.modules.nucleus2_moe_divfree import (
    Nucleus2MoEDivFreeTraining,
    Nucleus2MoEDivFreeTrainingSettings,
)
from nucleus.training.run import run_training


def build_training_module(model, cfg: DictConfig) -> Nucleus2MoEDivFreeTraining:
    settings = Nucleus2MoEDivFreeTrainingSettings(
        target_timesteps=cfg.data.target_timesteps,
        max_unrolls=cfg.data.unroll_steps - 1,
        **OmegaConf.to_container(cfg.training, resolve=True),
    )
    return Nucleus2MoEDivFreeTraining(model, settings)


@hydra.main(version_base=None, config_path="../../config", config_name="train_nucleus2_moe_divfree")
def main(cfg: DictConfig) -> None:
    run_training(cfg, build_training_module)


if __name__ == "__main__":
    main()
