from .nucleus1_moe import Nucleus1ViTMoE, Nucleus1AxialMoE, Nucleus1NeighborMoE, Nucleus1MoEConfig
from .nucleus1_vit import Nucleus1ViT, Nucleus1AxialViT, Nucleus1NeighborViT, Nucleus1ViTConfig
from .nucleus2_moe import Nucleus2MoE, Nucleus2MoEConfig
from .nucleus2_moe_divfree import Nucleus2MoEDivFree, Nucleus2MoEDivFreeConfig
from .nucleus2_moe_phase import Nucleus2MoEPhase
from .unets import ModernUnet, ClassicUnet
from .bubbleformer_vit import BubbleformerViT, BubbleformerFilmViT
from nucleus.baseline.moe_dpot import MoEPOTNet as MoEPOTNetModule
from nucleus.data.batching import CollatedBatch
from ._api import (
    register_model,
    list_models,
    get_model,
    get_model_class,
    load_model_from_checkpoint,
)

# This is just a stupid wrapper to register the model and initialize with a dictionary.
try:
    import transformers
    HAS_TRANSFORMERS_DEPS = True
except ImportError:
    HAS_TRANSFORMERS_DEPS = False
if HAS_TRANSFORMERS_DEPS:
    from nucleus.baseline.poseidon import ScOT, ScOTConfig

    @register_model("poseidon")
    class Poseidon(ScOT):
        def __init__(self, *args, **kwargs):
            super().__init__(ScOTConfig(**kwargs))

@register_model("moe_dpot")
class MoEPOTNet(MoEPOTNetModule):
    def __init__(self, *args, **kwargs):
        super().__init__(config=kwargs, router_loss_weight=kwargs.get("router_loss_weight"), lr=kwargs.get("lr"))