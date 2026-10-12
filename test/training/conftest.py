import pytest
import torch

from nucleus.models import get_model
from nucleus.models.nucleus2_moe import Nucleus2MoE
from nucleus.testing.flashx import flashx_normalizer
from nucleus.testing.reference_attention import reference_natten
from nucleus.training.modules.nucleus2_moe import Nucleus2MoETraining, Nucleus2MoETrainingSettings


@pytest.fixture(autouse=True)
def differentiable_attention_on_cpu():
    with reference_natten():
        yield


@pytest.fixture
def moe_training():
    torch.manual_seed(0)
    normalizer = flashx_normalizer(Nucleus2MoE.field_names, vector_fields=[("velx", "vely")])
    model = get_model(
        "nucleus2_moe", normalizer, patch_size=4, embed_dim=64, num_heads=2, processor_blocks=1,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
    )
    return Nucleus2MoETraining(model, Nucleus2MoETrainingSettings(target_timesteps=2))
