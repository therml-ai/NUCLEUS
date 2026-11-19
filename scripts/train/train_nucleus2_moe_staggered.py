import hydra
from omegaconf import DictConfig, OmegaConf

from nucleus.training.modules.nucleus2_moe_staggered import (
    Nucleus2MoEStaggeredTraining,
    Nucleus2MoEStaggeredTrainingSettings,
)
from nucleus.training.run import run_training


def build_training_module(model, cfg: DictConfig) -> Nucleus2MoEStaggeredTraining:
    settings = Nucleus2MoEStaggeredTrainingSettings(
        target_timesteps=cfg.data.target_timesteps,
        max_unrolls=cfg.data.unroll_steps - 1,
        **OmegaConf.to_container(cfg.training, resolve=True),
    )
    return Nucleus2MoEStaggeredTraining(model, settings)


@hydra.main(version_base=None, config_path="../../config", config_name="train_nucleus2_moe_staggered")
def main(cfg: DictConfig) -> None:
    run_training(cfg, build_training_module)


if __name__ == "__main__":
    main()
