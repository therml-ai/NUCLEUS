import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from diffusion import train_diffusion
from nucleus.utils.set_fp32_precision import set_fp32_precision

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    set_fp32_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_diffusion(cfg)


if __name__ == "__main__":
    main()
