import dataclasses
from pathlib import Path

import torch

from nucleus.models.legacy_nucleus2_moe import LegacyNucleus2MoE, Nucleus2MoEConfig


def load_legacy_diffusion_model(
    path: str | Path, model_cfg: dict
) -> LegacyNucleus2MoE:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in checkpoint:
        state = {
            key.removeprefix("model."): value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith("model.")
        }
    else:
        state = checkpoint
    extra_state = state.get("_extra_state")
    model_name = extra_state["model_name"] if extra_state is not None else model_cfg["name"]
    if model_name != "nucleus2_moe" or model_name != model_cfg["name"]:
        raise ValueError(f"Unsupported legacy diffusion surrogate {model_name!r}")
    if extra_state is None:
        fields = {field.name for field in dataclasses.fields(Nucleus2MoEConfig)}
        params = {key: value for key, value in model_cfg["params"].items() if key in fields}
        config = Nucleus2MoEConfig(**params)
        state = {
            **state,
            "_extra_state": {"model_name": model_name, "config": dataclasses.asdict(config)},
        }
    else:
        config = LegacyNucleus2MoE.config_from_dict(extra_state["config"])
    model = LegacyNucleus2MoE(config)
    model.load_state_dict(state)
    return model
