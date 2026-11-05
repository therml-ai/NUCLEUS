from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper

from nucleus.models.flashx_io import num_cell_channels, require_normalized
from nucleus.models.nucleus2_moe import Nucleus2MoEOutput
from nucleus.models.nucleus2_moe_base import Nucleus2MoEBase, Nucleus2MoEBaseConfig, config_from_dict, patch_debed

from ._api import register_model

__all__ = ["Nucleus2MoEContinuityLoss", "Nucleus2MoEContinuityLossConfig"]

Nucleus2MoEStaggeredConfig = Nucleus2MoEBaseConfig
Nucleus2MoEContinuityLossConfig = Nucleus2MoEBaseConfig


def _config_from_dict(config_dict: dict) -> Nucleus2MoEStaggeredConfig:
    return config_from_dict(Nucleus2MoEStaggeredConfig, config_dict)


@register_model("nucleus2_moe_continuity_loss")
class Nucleus2MoEContinuityLoss(Nucleus2MoEBase):
    """Predicts dfun, temperature and the face velocities on their natural grids, but
    does not enforce any physics.
    """
    config_class = Nucleus2MoEStaggeredConfig
    config_from_dict = staticmethod(_config_from_dict)
    field_names = ("dfun", "temperature", "velfacex", "velfacey")
    vector_fields = (("velfacex", "velfacey"),)

    def __init__(self, config: Nucleus2MoEStaggeredConfig, normalizer: NormalizerWrapper):
        # the prediction is laid out on the cells like the input
        channels = num_cell_channels(self.field_names)
        super().__init__(config, normalizer, in_channels=channels)
        self.debed = patch_debed(config, out_channels=channels)

    def forward(self, batch: FlashXBatch) -> Nucleus2MoEOutput:
        """batch: normalized, holding field_names and a config tensor of config_keys."""
        require_normalized(batch)
        cells, moe_outputs = self.step(batch.stack_field_cells(self.field_names), batch.config_tensor())
        return Nucleus2MoEOutput(batch.from_stacked_field_cells(cells, self.field_names), moe_outputs)

    def velocity_scale(self) -> float:
        offset_x, scale = self.normalizer.field_offset_and_scale("velfacex")
        offset_y, scale_y = self.normalizer.field_offset_and_scale("velfacey")
        if (offset_x, offset_y) != (0.0, 0.0) or scale != scale_y:
            raise ValueError("velfacex and velfacey must be normalized as one of the normalizer's vector_fields")
        return scale
