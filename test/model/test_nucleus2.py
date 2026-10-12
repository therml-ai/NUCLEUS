import pytest
import torch

from nucleus.models import get_model, load_model
from nucleus.models.nucleus2_moe import Nucleus2MoE
from nucleus.testing.flashx import flashx_normalizer, random_batch
from nucleus.testing.reference_attention import reference_natten

SMALL_MODEL = dict(
    patch_size=4, embed_dim=64, num_heads=2, processor_blocks=2,
    num_experts=4, topk=2, moe_intermediate_dim=64, patching="Linear",
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    normalizer = flashx_normalizer(Nucleus2MoE.field_names, vector_fields=[("velx", "vely")])
    return get_model("nucleus2_moe", normalizer, **SMALL_MODEL)


def _batch(model, batch_size=2):
    return random_batch(Nucleus2MoE.field_names, Nucleus2MoE.config_keys, batch_size=batch_size)


def test_forward_predicts_every_field_for_the_same_samples(model):
    batch = model.normalizer.normalize(_batch(model))
    with torch.no_grad():
        output = model(batch)
    assert set(output.prediction.fields) == set(Nucleus2MoE.field_names)
    for name, tensor in output.prediction.fields.items():
        assert tensor.shape == batch.fields[name].shape
        assert torch.isfinite(tensor).all()
    assert output.prediction.is_normalized
    assert output.prediction.configs == batch.configs
    assert len(output.moe_outputs) == SMALL_MODEL["processor_blocks"]


def test_forward_needs_a_normalized_batch(model):
    with pytest.raises(ValueError, match="normalized batches"):
        model(_batch(model))


def test_gradients_are_finite(model):
    batch = model.normalizer.normalize(_batch(model))
    with reference_natten():
        output = model(batch)
    output.prediction.stacked_fields(Nucleus2MoE.field_names).sum().backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_predict_next_gives_raw_frames_that_extend_the_window(model):
    window = _batch(model)
    prediction = model.predict_next(window, num_timesteps=1)
    assert not prediction.is_normalized
    assert prediction.num_timesteps == 1
    assert window.extend(prediction).num_timesteps == window.num_timesteps + 1


def test_load_model_rebuilds_the_model_and_its_normalizer(model, tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, path)
    loaded = load_model(path)
    assert isinstance(loaded, Nucleus2MoE)
    assert loaded.normalizer.statistics == model.normalizer.statistics
    assert loaded.normalizer.vector_fields == [["velx", "vely"]]
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, torch.Tensor):
            torch.testing.assert_close(loaded.state_dict()[name], tensor)
