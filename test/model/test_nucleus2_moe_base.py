import pytest
import torch

from nucleus.models.nucleus2_moe_base import (
    AdaptiveLayerNorm,
    Nucleus2MoEBase,
    Nucleus2MoEBaseConfig,
    patch_debed,
)
from nucleus.testing.flashx import flashx_normalizer

IN_CHANNELS, OUT_CHANNELS = 3, 5


class _Head(Nucleus2MoEBase):
    def __init__(self, config, normalizer):
        super().__init__(config, normalizer, in_channels=IN_CHANNELS)
        self.debed = patch_debed(config, out_channels=OUT_CHANNELS)

    def step(self, cells, sim_params):
        x, moe_outputs = self.process(cells, sim_params)
        return self.debed(x, target_shape=cells.shape[2:4]), moe_outputs


@pytest.mark.parametrize("patching", ["Linear", "Adaptive"])
def test_a_derived_model_predicts_its_own_number_of_channels(patching):
    torch.manual_seed(0)
    config = Nucleus2MoEBaseConfig(
        patch_size=4, embed_dim=64, num_heads=2, processor_blocks=2,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching=patching,
    )
    model = _Head(config, flashx_normalizer(("dfun",)))
    cells = torch.rand(2, 2, 32, 32, IN_CHANNELS)
    sim_params = torch.rand(2, model.num_sim_params)
    with torch.no_grad():
        output, moe_outputs = model.step(cells, sim_params)
    assert output.shape == (2, 2, 32, 32, OUT_CHANNELS)
    assert len(moe_outputs) == config.processor_blocks


SMALL_CONFIG = dict(
    patch_size=4, embed_dim=64, num_heads=2, processor_blocks=1,
    num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
)


def test_every_block_modulates_from_the_embedded_sim_params():
    torch.manual_seed(0)
    config = Nucleus2MoEBaseConfig(**SMALL_CONFIG)
    model = _Head(config, flashx_normalizer(("dfun",)))
    for block in model.blocks:
        assert block.attention_norm.modulation.in_features == config.embed_dim
    assert model.sim_params_embed(torch.rand(3, model.num_sim_params)).shape == (3, config.embed_dim)


def test_the_conditioning_depends_nonlinearly_on_the_sim_params():
    torch.manual_seed(0)
    model = _Head(Nucleus2MoEBaseConfig(**SMALL_CONFIG), flashx_normalizer(("dfun",)))
    direction = torch.randn(model.num_sim_params)
    low, middle, high = model.sim_params_embed(torch.stack([0.1 * direction, 0.5 * direction, 0.9 * direction]))
    # a map affine in the sim params would put the middle run halfway between the others
    assert not torch.allclose(middle, (low + high) / 2, atol=1e-4)


NORM_EMBED_DIM, CONDITIONING_DIM = 16, 8


def _tokens(batch_size: int) -> torch.Tensor:
    return torch.ones(batch_size, 4, NORM_EMBED_DIM).cumsum(-1)


def test_a_new_layer_is_a_plain_layer_norm():
    norm = AdaptiveLayerNorm(NORM_EMBED_DIM, CONDITIONING_DIM, torch.float32)
    tokens = _tokens(2)
    torch.testing.assert_close(
        norm(tokens, torch.randn(2, CONDITIONING_DIM)), torch.nn.functional.layer_norm(tokens, (NORM_EMBED_DIM,), eps=1e-6)
    )


def test_the_modulation_tells_apart_conditionings_along_one_direction():
    torch.manual_seed(0)
    norm = AdaptiveLayerNorm(NORM_EMBED_DIM, CONDITIONING_DIM, torch.float32)
    with torch.no_grad():
        norm.modulation.weight.normal_()
        # a sweep scales one direction of the conditioning, which a LayerNorm over it
        # would reduce to its sign
        conditioning = torch.tensor([[0.1], [0.5], [0.9]]) * torch.randn(CONDITIONING_DIM)
        modulated = norm(_tokens(3), conditioning)
    assert not torch.allclose(modulated[0], modulated[1]) and not torch.allclose(modulated[1], modulated[2])
