from dataclasses import dataclass
from typing import List

from nucleus.layers.moe.topk_moe import TopkMoEOutput
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper
from nucleus.models.flashx_io import require_normalized
from nucleus.models.nucleus2_moe_base import Nucleus2MoEBase, Nucleus2MoEBaseConfig, config_from_dict, patch_debed

from ._api import register_model

__all__ = ["Nucleus2MoE", "Nucleus2MoEConfig", "Nucleus2MoEOutput"]


Nucleus2MoEConfig = Nucleus2MoEBaseConfig


def _config_from_dict(config_dict: dict) -> Nucleus2MoEConfig:
    return config_from_dict(Nucleus2MoEConfig, config_dict)


@dataclass
class Nucleus2MoEOutput:
    prediction: FlashXBatch  # normalized; every frame of the output window
    moe_outputs: List[TopkMoEOutput]


@register_model("nucleus2_moe")
class Nucleus2MoE(Nucleus2MoEBase):
    config_class = Nucleus2MoEConfig
    config_from_dict = staticmethod(_config_from_dict)
    field_names = ("dfun", "temperature", "velx", "vely")

    def __init__(self, config: Nucleus2MoEConfig, normalizer: NormalizerWrapper):
        super().__init__(config, normalizer, in_channels=len(self.field_names))
        self.debed = patch_debed(config, out_channels=len(self.field_names))

    def forward(self, batch: FlashXBatch) -> Nucleus2MoEOutput:
        """batch: normalized, holding field_names and a config tensor of config_keys."""
        require_normalized(batch)
        output, moe_outputs = self.step(batch.stacked_fields(self.field_names), batch.config_tensor())
        prediction = batch.with_fields(dict(zip(self.field_names, output.unbind(-1))))
        return Nucleus2MoEOutput(prediction, moe_outputs)
