from .vit import ViT, AxialViT, NeighborViT
from .moe import NeighborMoE
from .nucleus1_moe import Nucleus1ViTMoE, Nucleus1AxialMoE, Nucleus1NeighborMoE
from .nucleus1_vit import Nucleus1ViT, Nucleus1AxialViT, Nucleus1NeighborViT
from .unets import ModernUnet, ClassicUnet
from .bubbleformer_vit import BubbleformerViT, BubbleformerFilmViT
from nucleus.baseline.poseidon import ScOT, ScOTConfig
from nucleus.baseline.moe_dpot import MoEPOTNet
from nucleus.data.batching import CollatedBatch
from ._api import (
    register_model,
    list_models,
    get_model
)

# This is just a stupid wrapper to register the model and initialize with a dictionary.
@register_model("poseidon")
class Poseidon(ScOT):
    def __init__(self, *args, **kwargs):
        super().__init__(ScOTConfig(**kwargs))

@register_model("moe_dpot")
class MoEPOTNet(MoEPOTNet):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)