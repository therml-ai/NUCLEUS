import pytest
import torch

from nucleus.models.nucleus2_moe_base import Nucleus2MoEBase, Nucleus2MoEBaseConfig, patch_debed
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
