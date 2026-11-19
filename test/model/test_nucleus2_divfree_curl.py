import pytest
import torch
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper

from nucleus.models import get_model, load_model
from nucleus.models.nucleus2_moe_divfree_curl import (
    CORNER_OFFSETS,
    Nucleus2MoEDivFreeCurl,
    nodes_from_cell_corners,
)
from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.testing.flashx import flashx_statistics, random_batch
from nucleus.testing.reference_attention import reference_natten

FIELD_NAMES = Nucleus2MoEDivFreeCurl.field_names
SMALL_MODEL = dict(
    patch_size=4, embed_dim=64, num_heads=2, processor_blocks=2,
    num_experts=4, topk=2, moe_intermediate_dim=64, patching="Linear",
)
# drop the top outflow row, where the Dirichlet condition breaks div = source
INTERIOR = (slice(None), slice(None), slice(None, -1), slice(None))


def _model(statistics=None, **config) -> Nucleus2MoEDivFreeCurl:
    torch.manual_seed(0)
    normalizer = NormalizerWrapper(
        statistics or flashx_statistics(FIELD_NAMES), vector_fields=Nucleus2MoEDivFreeCurl.vector_fields
    )
    return get_model("nucleus2_moe_divfree_curl", normalizer, **SMALL_MODEL, **config)


@pytest.fixture
def model() -> Nucleus2MoEDivFreeCurl:
    model = _model()
    with torch.no_grad():
        # a nonzero head makes the predicted sdf span an interface and the stream
        # function and divergence source nonzero
        model.debed.linear.weight.normal_(std=0.3)
    return model


def _normalized_batch(model, batch_size=1, height=32, width=32) -> FlashXBatch:
    batch = random_batch(FIELD_NAMES, model.config_keys, batch_size=batch_size, height=height, width=width)
    return model.normalizer.normalize(batch)


def _run(model, batch, **options):
    with torch.no_grad():
        return model(batch, **options)


def _divergence(velx, vely, grid):
    return divergence_centers_from_faces(velx.double(), vely.double(), grid.dx, grid.dy)


def test_forward_predicts_every_field_on_its_natural_grid(model):
    batch = _normalized_batch(model, batch_size=2)
    output = _run(model, batch)
    for name in FIELD_NAMES:
        assert output.prediction.fields[name].shape == batch.fields[name].shape
        assert output.prediction.fields[name].dtype == torch.float32
        assert torch.isfinite(output.prediction.fields[name]).all()
    assert output.prediction.is_normalized
    batch_size, timesteps, height, width = batch.fields["dfun"].shape
    assert output.psi_nodes.shape == (batch_size, timesteps, height + 1, width + 1)
    assert output.div_source.shape == batch.fields["dfun"].shape


def test_the_velocity_is_the_solenoidal_plus_the_dilatational_velocity(model):
    output = _run(model, _normalized_batch(model))
    torch.testing.assert_close(
        output.prediction.fields["velfacex"], output.solenoidal_velx_faces + output.dilatational_velx_faces
    )
    torch.testing.assert_close(
        output.prediction.fields["velfacey"], output.solenoidal_vely_faces + output.dilatational_vely_faces
    )


def test_the_solenoidal_velocity_is_divergence_free(model):
    output = _run(model, _normalized_batch(model))
    assert output.solenoidal_velx_faces.abs().max() > 0.1
    divergence = _divergence(output.solenoidal_velx_faces, output.solenoidal_vely_faces, output.grid)
    assert divergence.abs().max() < 1e-3


@pytest.mark.parametrize("use_mass_transfer", [False, True])
def test_the_velocity_diverges_as_the_gated_source(model, use_mass_transfer):
    output = _run(model, _normalized_batch(model), use_div_gate=True, use_mass_transfer=use_mass_transfer)
    velocity = output.prediction.fields
    torch.testing.assert_close(
        _divergence(velocity["velfacex"], velocity["velfacey"], output.grid)[INTERIOR],
        output.gated_div_source[INTERIOR].double(),
        atol=1e-3, rtol=0,
    )


def test_div_gate_confines_the_source_to_the_interface_band(model):
    output = _run(model, _normalized_batch(model), use_div_gate=True)
    sdf = model.normalizer.unnormalize_field("dfun", output.prediction.fields["dfun"])
    outside = sdf.abs() > 3 * output.grid.dx
    assert outside.any() and output.div_source[outside].abs().max() > 0
    assert output.gated_div_source[outside].abs().max() == 0


def test_mass_transfer_replaces_the_learned_source(model):
    batch = _normalized_batch(model)
    learned = _run(model, batch)
    physics = _run(model, batch, use_mass_transfer=True)
    assert torch.isfinite(physics.div_source).all()
    assert not torch.allclose(physics.div_source, learned.div_source)
    torch.testing.assert_close(physics.gated_div_source, physics.div_source)


@pytest.mark.parametrize("use_mass_transfer", [False, True])
def test_forward_needs_velocities_normalized_as_one_vector(use_mass_transfer):
    statistics = flashx_statistics(FIELD_NAMES)
    statistics["fields"]["velfacey"]["std"] = 3.0
    model = _model(statistics)
    model.normalizer.vector_fields = []
    with pytest.raises(ValueError, match="vector_fields"):
        _run(model, _normalized_batch(model), use_mass_transfer=use_mass_transfer)


def test_gradients_are_finite(model):
    with reference_natten():
        output = model(_normalized_batch(model))
    sum(tensor.sum() for tensor in output.prediction.fields.values()).backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_predict_next_gives_raw_frames_that_extend_the_window(model):
    window = random_batch(FIELD_NAMES, model.config_keys)
    prediction = model.predict_next(window, num_timesteps=1, target_timesteps=2)
    assert not prediction.is_normalized and prediction.num_timesteps == 1
    assert window.extend(prediction).num_timesteps == window.num_timesteps + 1


def test_a_checkpoint_round_trips(model, tmp_path):
    model.eval()
    torch.save({"model": model.state_dict()}, tmp_path / "checkpoint.pt")
    loaded = load_model(tmp_path / "checkpoint.pt").eval()
    assert isinstance(loaded, Nucleus2MoEDivFreeCurl)
    batch = _normalized_batch(model)
    torch.testing.assert_close(_run(loaded, batch).prediction.fields["velfacex"], _run(model, batch).prediction.fields["velfacex"])


def test_nodes_average_the_corners_of_the_cells_that_share_them():
    corners = torch.arange(2 * 3 * len(CORNER_OFFSETS), dtype=torch.float64).reshape(2, 3, len(CORNER_OFFSETS))
    nodes = nodes_from_cell_corners(corners)
    assert nodes.shape == (3, 4)
    # the bottom-left node is only cell (0, 0)'s bottom-left corner
    assert nodes[0, 0] == corners[0, 0, 0]
    # an interior node is the top-right, top-left, bottom-right and bottom-left
    # corner of the four cells around it
    expected = (corners[0, 0, 3] + corners[0, 1, 2] + corners[1, 0, 1] + corners[1, 1, 0]) / 4
    torch.testing.assert_close(nodes[1, 1], expected)


def test_consistent_corners_give_back_their_nodes():
    nodes = torch.randn(2, 5, 6, dtype=torch.float64)
    corners = torch.stack(
        [nodes[..., row:row + 4, column:column + 5] for row, column in CORNER_OFFSETS], dim=-1
    )
    torch.testing.assert_close(nodes_from_cell_corners(corners), nodes)
