import pytest
import torch

from nucleus.models.legacy_diffusion_checkpoint import load_legacy_diffusion_model
from nucleus.models.legacy_nucleus2_moe import LegacyNucleus2MoE, Nucleus2MoEConfig


@pytest.mark.parametrize("has_extra_state", [False, True])
def test_load_lightning_diffusion_surrogate(tmp_path, has_extra_state):
    params = {
        "patch_size": 4,
        "embed_dim": 32,
        "num_heads": 2,
        "processor_blocks": 1,
        "num_experts": 2,
        "topk": 1,
        "moe_intermediate_dim": 32,
        "patching": "Linear",
    }
    model = LegacyNucleus2MoE(Nucleus2MoEConfig(**params))
    model._model_name = "nucleus2_moe"
    state = model.state_dict()
    if not has_extra_state:
        del state["_extra_state"]
    path = tmp_path / "last.ckpt"
    torch.save({"state_dict": {f"model.{key}": value for key, value in state.items()}}, path)

    restored = load_legacy_diffusion_model(path, {"name": "nucleus2_moe", "params": params})

    assert restored._model_name == "nucleus2_moe"
    torch.testing.assert_close(restored.embed.linear.weight, model.embed.linear.weight)
