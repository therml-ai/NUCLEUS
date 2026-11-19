import pytest
import torch
from boiling_data.flashx.batch import unstack_field_cells

from nucleus.models import get_model, load_model
from nucleus.models.nucleus2_moe_staggered import Nucleus2MoEStaggered
from nucleus.testing.flashx import flashx_normalizer, random_batch
from nucleus.testing.reference_attention import reference_natten

FIELD_NAMES = Nucleus2MoEStaggered.field_names
SMALL_MODEL = dict(
    patch_size=4, embed_dim=64, num_heads=2, processor_blocks=2,
    num_experts=4, topk=2, moe_intermediate_dim=64, patching="Linear",
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    normalizer = flashx_normalizer(FIELD_NAMES)
    return get_model("nucleus2_moe_staggered", normalizer, **SMALL_MODEL)


def _normalized_batch(model, batch_size=2):
    return model.normalizer.normalize(random_batch(FIELD_NAMES, model.config_keys, batch_size=batch_size))


def test_forward_predicts_every_field_on_its_own_grid(model):
    batch = _normalized_batch(model)
    with torch.no_grad():
        output = model(batch)
    assert set(output.prediction.fields) == set(FIELD_NAMES)
    for name in FIELD_NAMES:
        assert output.prediction.fields[name].shape == batch.fields[name].shape
        assert torch.isfinite(output.prediction.fields[name]).all()
    assert output.prediction.is_normalized
    assert len(output.moe_outputs) == SMALL_MODEL["processor_blocks"]


def test_the_prediction_is_the_network_output_with_no_physics_applied(model):
    model.eval()
    batch = _normalized_batch(model)
    with torch.no_grad():
        cells, _ = model.step(batch.stack_field_cells(FIELD_NAMES), batch.config_tensor())
        prediction = model(batch).prediction
    for name, tensor in unstack_field_cells(cells, FIELD_NAMES).items():
        torch.testing.assert_close(prediction.fields[name], tensor)


def test_gradients_reach_every_parameter_of_the_head(model):
    with reference_natten():
        output = model(_normalized_batch(model))
    sum(tensor.sum() for tensor in output.prediction.fields.values()).backward()
    assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.debed.parameters())


def test_predict_next_gives_raw_frames_that_extend_the_window(model):
    window = random_batch(FIELD_NAMES, model.config_keys, batch_size=1)
    prediction = model.predict_next(window, num_timesteps=1, target_timesteps=2)
    assert not prediction.is_normalized
    assert window.extend(prediction).num_timesteps == window.num_timesteps + 1


def test_load_model_rebuilds_the_model(model, tmp_path):
    torch.save({"model": model.state_dict()}, tmp_path / "checkpoint.pt")
    loaded = load_model(tmp_path / "checkpoint.pt")
    assert isinstance(loaded, Nucleus2MoEStaggered)
    for name, tensor in model.state_dict().items():
        if isinstance(tensor, torch.Tensor):
            torch.testing.assert_close(loaded.state_dict()[name], tensor)
