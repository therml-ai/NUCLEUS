import pytest
import torch
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper

from nucleus.models import get_model
from nucleus.models.nucleus2_moe_divfree import Nucleus2MoEDivFree
from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.testing.flashx import flashx_config, flashx_statistics, random_batch
from nucleus.testing.reference_attention import reference_natten

FIELD_NAMES = Nucleus2MoEDivFree.field_names
SMALL_MODEL = dict(
    patch_size=4, embed_dim=64, num_heads=2, processor_blocks=2,
    num_experts=4, topk=2, moe_intermediate_dim=64, patching="Linear",
)
# drop the top outflow row, where the Dirichlet condition breaks div = source
INTERIOR = (slice(None), slice(None), slice(None, -1), slice(None))


def _model(statistics=None) -> Nucleus2MoEDivFree:
    torch.manual_seed(0)
    normalizer = NormalizerWrapper(
        statistics or flashx_statistics(FIELD_NAMES), vector_fields=Nucleus2MoEDivFree.vector_fields
    )
    return get_model("nucleus2_moe_divfree", normalizer, **SMALL_MODEL)


@pytest.fixture
def model() -> Nucleus2MoEDivFree:
    return _model()


def _normalized_batch(model, batch_size=1, height=32, width=32) -> FlashXBatch:
    batch = random_batch(FIELD_NAMES, model.config_keys, batch_size=batch_size, height=height, width=width)
    return model.normalizer.normalize(batch)


def _run(model, batch, **options):
    with torch.no_grad():
        return model(batch, **options)


def test_forward_predicts_every_field_on_its_natural_grid(model):
    batch = _normalized_batch(model, batch_size=2)
    output = _run(model, batch)
    for name in FIELD_NAMES:
        assert output.prediction.fields[name].shape == batch.fields[name].shape
        assert torch.isfinite(output.prediction.fields[name]).all()
    assert output.prediction.is_normalized
    assert output.velx_sol.shape == batch.fields["velfacex"].shape
    assert output.phi.shape == batch.fields["dfun"].shape
    assert (output.grid.dx, output.grid.dy) == (pytest.approx(1 / 16), pytest.approx(1 / 16))


def test_forward_needs_a_normalized_batch(model):
    with pytest.raises(ValueError, match="normalized batches"):
        model(random_batch(FIELD_NAMES, model.config_keys))


def test_gradients_are_finite(model):
    with reference_natten():
        output = model(_normalized_batch(model))
    sum(tensor.sum() for tensor in output.prediction.fields.values()).backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def test_temperature_ansatz_uses_the_normalized_saturation_temperature():
    # a temperature normalization far from the identity, so mixing raw and
    # normalized saturation temperatures gives a detectably wrong value
    statistics = flashx_statistics(FIELD_NAMES)
    statistics["fields"]["temperature"] |= {"mean": 0.3, "std": 2.0}
    model = _model(statistics)
    with torch.no_grad():
        # the debed's linear layer has no bias, so a zero weight makes every head
        # output zero: the sdf sits on the interface and the network adds nothing
        model.debed.linear.weight.zero_()
    output = _run(model, _normalized_batch(model))

    # flashx_config: (sat_temp - bulk_temp) / (wall_temp_scale - bulk_temp) = 8 / 20
    expected = (0.4 - 0.3) / 2.0
    # the last column is past the heater's x extent, out of its band
    far_from_heater = output.prediction.fields["temperature"][..., -1]
    torch.testing.assert_close(far_from_heater, torch.full_like(far_from_heater, expected), atol=1e-4, rtol=0)


def test_each_sample_uses_its_own_parameters(model):
    with torch.no_grad():
        # a zero head leaves the temperature to the ansatz's parameters alone; the
        # network's bfloat16 attention and MoE need not give one sample the same
        # values in batches of different sizes
        model.debed.linear.weight.zero_()
    batch = _normalized_batch(model, batch_size=2)
    batched = _run(model, batch).prediction.fields["temperature"]
    for index in range(2):
        alone = FlashXBatch(
            {name: tensor[index:index + 1] for name, tensor in batch.fields.items()},
            [batch.configs[index]], batch.config_keys, is_normalized=True,
        )
        torch.testing.assert_close(batched[index], _run(model, alone).prediction.fields["temperature"][0])
    # random_batch gives the samples different heater temperatures
    assert not torch.allclose(batched[0], batched[1])


def test_div_projection_makes_the_solenoidal_part_divergence_free(model):
    with torch.no_grad():
        model.debed.linear.weight.normal_()
    output = _run(model, _normalized_batch(model), use_div_projection=True)
    divergence = divergence_centers_from_faces(output.velx_sol, output.vely_sol, output.grid.dx, output.grid.dy)
    assert divergence[INTERIOR].abs().max() < 1e-2


def test_div_gate_confines_divergence_to_the_interface_band(model):
    with torch.no_grad():
        # a nonzero debed makes the predicted sdf span an interface and the
        # divergence source nonzero
        model.debed.linear.weight.normal_(std=0.3)
    batch = _normalized_batch(model)
    gated = _run(model, batch, use_div_projection=True, use_div_gate=True)
    ungated = _run(model, batch, use_div_projection=True, use_div_gate=False)

    sdf = model.normalizer.unnormalize_field("dfun", gated.prediction.fields["dfun"])
    in_band = (sdf.abs() <= 5 * gated.grid.dx)[INTERIOR]
    assert in_band.any() and (~in_band).any()

    def divergence(output):
        fields = output.prediction.fields
        return divergence_centers_from_faces(fields["velfacex"], fields["velfacey"], output.grid.dx, output.grid.dy)[INTERIOR]

    assert divergence(gated)[~in_band].abs().max() < 1e-2
    assert divergence(gated)[in_band].abs().max() > 1e-1
    assert divergence(ungated)[~in_band].abs().max() > 1e-1


def test_mass_transfer_replaces_the_learned_source(model):
    batch = _normalized_batch(model)
    learned = _run(model, batch)
    physics = _run(model, batch, use_mass_transfer=True)
    assert physics.div_source.shape == learned.div_source.shape
    for tensor in (*physics.prediction.fields.values(), physics.div_source):
        assert torch.isfinite(tensor).all()
    assert not torch.allclose(physics.div_source, learned.div_source)


def test_mass_transfer_needs_velocities_normalized_as_one_vector():
    statistics = flashx_statistics(FIELD_NAMES)
    statistics["fields"]["velfacex"]["mean"] = 0.5
    model = _model(statistics)
    model.normalizer.vector_fields = []
    with pytest.raises(ValueError, match="vector_fields"):
        _run(model, _normalized_batch(model), use_mass_transfer=True)


def test_grid_spacing_comes_from_the_discretization(model):
    # 32 cells over twice the default domain
    config = flashx_config(width=64, height=64)
    batch = random_batch(FIELD_NAMES, model.config_keys, batch_size=1)
    batch = FlashXBatch(dict(batch.fields), [config], batch.config_keys)
    output = _run(model, model.normalizer.normalize(batch))
    assert (output.grid.dx, output.grid.dy) == (pytest.approx(1 / 8), pytest.approx(1 / 8))


def test_a_batch_mixing_grid_spacings_is_rejected(model):
    batch = random_batch(FIELD_NAMES, model.config_keys, batch_size=2)
    batch = FlashXBatch(dict(batch.fields), [flashx_config(), flashx_config(width=64)], batch.config_keys)
    with pytest.raises(ValueError, match="one grid spacing"):
        _run(model, model.normalizer.normalize(batch))


def test_predict_next_gives_raw_frames_that_extend_the_window(model):
    window = random_batch(FIELD_NAMES, model.config_keys)
    prediction = model.predict_next(window, num_timesteps=1)
    assert not prediction.is_normalized and prediction.num_timesteps == 1
    assert prediction.fields["velfacey"].shape[-2:] == window.fields["velfacey"].shape[-2:]
    assert window.extend(prediction).num_timesteps == window.num_timesteps + 1
