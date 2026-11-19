import pytest
import torch
from boiling_data.flashx.batch import FlashXBatch
from boiling_data.flashx.normalizer import NormalizerWrapper

from nucleus.models import get_model, load_model
from nucleus.models.flashx_io import set_runtime_options
from nucleus.models.nucleus2_moe_divfree_projection import Nucleus2MoEDivFreeProjection, divfree_physics, heater_mask
from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.testing.flashx import flashx_config, flashx_statistics, random_batch
from nucleus.testing.reference_attention import reference_natten

FIELD_NAMES = Nucleus2MoEDivFreeProjection.field_names
SMALL_MODEL = dict(
    patch_size=4, embed_dim=64, num_heads=2, processor_blocks=2,
    num_experts=4, topk=2, moe_intermediate_dim=64, patching="Linear",
)
# drop the top outflow row, where the Dirichlet condition breaks div = source
INTERIOR = (slice(None), slice(None), slice(None, -1), slice(None))


def _model(statistics=None, **config) -> Nucleus2MoEDivFreeProjection:
    torch.manual_seed(0)
    normalizer = NormalizerWrapper(
        statistics or flashx_statistics(FIELD_NAMES), vector_fields=Nucleus2MoEDivFreeProjection.vector_fields
    )
    return get_model("nucleus2_moe_divfree_projection", normalizer, **SMALL_MODEL, **config)


@pytest.fixture
def model() -> Nucleus2MoEDivFreeProjection:
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
    assert output.nn_velx_df.shape == output.velx_div_proj.shape == batch.fields["velfacex"].shape
    assert output.div_source.shape == batch.fields["dfun"].shape
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
    model = _model(statistics, use_temperature_ansatz=True)
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


def test_without_the_temperature_ansatz_the_temperature_is_the_networks(model):
    with torch.no_grad():
        # a zero head makes the network's temperature zero, where the ansatz would put
        # the saturation and heater temperatures
        model.debed.linear.weight.zero_()
    temperature = _run(model, _normalized_batch(model)).prediction.fields["temperature"]
    torch.testing.assert_close(temperature, torch.zeros_like(temperature))


def test_each_sample_uses_its_own_parameters():
    model = _model(use_temperature_ansatz=True)
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


@pytest.mark.parametrize("use_div_projection", [True, False])
def test_the_prediction_holds_the_requested_velocities(model, use_div_projection):
    with torch.no_grad():
        model.debed.linear.weight.normal_()
    output = _run(model, _normalized_batch(model), use_div_projection=use_div_projection)
    velx, vely = (output.velx_div_proj, output.vely_div_proj) if use_div_projection else (output.nn_velx_df, output.nn_vely_df)
    assert torch.equal(output.prediction.fields["velfacex"], velx)
    assert torch.equal(output.prediction.fields["velfacey"], vely)
    assert not torch.allclose(output.velx_div_proj, output.nn_velx_df)


def test_div_gate_confines_divergence_to_the_interface_band(model):
    with torch.no_grad():
        # a nonzero debed makes the predicted sdf span an interface and the
        # divergence source nonzero
        model.debed.linear.weight.normal_(std=0.3)
    batch = _normalized_batch(model)
    gated = _run(model, batch, use_div_gate=True)
    ungated = _run(model, batch, use_div_gate=False)

    sdf = model.normalizer.unnormalize_field("dfun", gated.prediction.fields["dfun"])
    in_band = (sdf.abs() <= 5 * gated.grid.dx)[INTERIOR]
    assert in_band.any() and (~in_band).any()

    def divergence(output):
        return _projected_divergence(output)[INTERIOR]

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
    prediction = model.predict_next(window, num_timesteps=1, target_timesteps=2)
    assert not prediction.is_normalized and prediction.num_timesteps == 1
    assert prediction.fields["velfacey"].shape[-2:] == window.fields["velfacey"].shape[-2:]
    assert window.extend(prediction).num_timesteps == window.num_timesteps + 1


def _projected_divergence(output):
    return divergence_centers_from_faces(output.velx_div_proj, output.vely_div_proj, output.grid.dx, output.grid.dy)


@pytest.mark.parametrize("use_mass_transfer", [False, True])
def test_the_projected_velocity_diverges_as_the_gated_source(model, use_mass_transfer):
    with torch.no_grad():
        model.debed.linear.weight.normal_(std=0.3)
    output = _run(model, _normalized_batch(model), use_div_gate=True, use_mass_transfer=use_mass_transfer)
    torch.testing.assert_close(
        _projected_divergence(output)[INTERIOR], output.gated_div_source[INTERIOR], atol=1e-3, rtol=0
    )


def test_the_config_sets_the_physics_a_call_does_not(model):
    # eval, so drop-path does not make the compared runs differ
    model.eval()
    with torch.no_grad():
        model.debed.linear.weight.normal_(std=0.3)
    batch = _normalized_batch(model)
    model.config.use_mass_transfer = True
    torch.testing.assert_close(_run(model, batch).div_source, _run(model, batch, use_mass_transfer=True).div_source)
    assert not torch.allclose(_run(model, batch, use_mass_transfer=False).div_source, _run(model, batch).div_source)


@pytest.mark.parametrize("use_div_projection", [True, False])
def test_rollouts_use_the_velocity_the_config_asks_for(model, use_div_projection):
    # eval, so drop-path does not make the compared runs differ
    model.eval()
    with torch.no_grad():
        model.debed.linear.weight.normal_(std=0.3)
    model.config.use_div_projection = use_div_projection
    window = random_batch(FIELD_NAMES, model.config_keys)
    output = _run(model, model.normalizer.normalize(window))
    expected = output.velx_div_proj if use_div_projection else output.nn_velx_df
    predicted = model.predict_next(window, num_timesteps=1, target_timesteps=2).fields["velfacex"]
    # the first frame of the two after the input
    torch.testing.assert_close(predicted, model.normalizer.unnormalize_field("velfacex", expected[:, :1]))


def test_the_continuity_source_is_not_gated(model):
    with torch.no_grad():
        model.debed.linear.weight.normal_(std=0.3)
    output = _run(model, _normalized_batch(model), use_div_gate=True, use_mass_transfer=True)
    torch.testing.assert_close(output.gated_div_source, output.div_source)


def test_loading_a_checkpoint_keeps_the_runtime_options_the_model_was_built_with():
    saved = _model(use_div_projection=False, use_mass_transfer=False)
    resumed = _model(use_div_projection=True, use_mass_transfer=True)
    resumed.load_state_dict(saved.state_dict())
    assert resumed.config.use_div_projection and resumed.config.use_mass_transfer
    for name, tensor in saved.state_dict().items():
        if isinstance(tensor, torch.Tensor):
            torch.testing.assert_close(resumed.state_dict()[name], tensor)


def test_runtime_options_are_set_from_params_and_the_architecture_kept(model):
    set_runtime_options(model, {"use_div_projection": True, "embed_dim": 512, "processor_blocks": 12})
    assert model.config.use_div_projection
    assert (model.config.embed_dim, model.config.processor_blocks) == (64, 2)


def test_the_continuity_source_sends_no_gradient_to_the_sdf(model):
    with torch.no_grad():
        model.debed.linear.weight.normal_(std=0.3)
    with reference_natten():
        output = model(_normalized_batch(model), use_mass_transfer=True)
    output.div_source.sum().backward()
    # the debed's output rows are laid out channel first: sdf, then temperature
    rows_per_channel = SMALL_MODEL["patch_size"] ** 2
    grad = model.debed.linear.weight.grad
    assert torch.count_nonzero(grad[:rows_per_channel]) == 0
    assert torch.count_nonzero(grad[rows_per_channel:2 * rows_per_channel]) > 0


def test_the_heater_mask_marks_the_cells_over_the_heater(model):
    # flashx_config's heater spans x in [-0.5, 0.5] on 32 cells over [-1, 1]
    physics = divfree_physics(random_batch(FIELD_NAMES, model.config_keys, batch_size=1, height=32, width=32))
    mask = heater_mask(physics)[0, 0, 0]
    x_centers = physics.grid.x_centers[0, 0, 0]
    assert mask.sum() == 16
    assert torch.equal(mask, (x_centers > -0.5) & (x_centers < 0.5))


def _faces_divergence(velx, vely, grid):
    return divergence_centers_from_faces(velx, vely, grid.dx, grid.dy)


def test_velocity_boundaries_set_the_walls_and_the_outflow(model):
    with torch.no_grad():
        model.debed.linear.weight.normal_(std=0.3)
    batch = _normalized_batch(model)
    free = _run(model, batch)
    bounded = _run(model, batch, use_velocity_boundaries=True)
    # flashx_config: no-slip left, right and bottom walls, outflow at the top
    assert free.nn_velx_df[..., 0].abs().max() > 0 and free.nn_vely_df[..., 0, :].abs().max() > 0
    for velocities in ((bounded.nn_velx_df, bounded.nn_vely_df), (bounded.velx_div_proj, bounded.vely_div_proj)):
        velx, vely = velocities
        assert velx[..., 0].abs().max() == 0 and velx[..., -1].abs().max() == 0
        assert vely[..., 0, :].abs().max() == 0
    top_cells = _faces_divergence(bounded.nn_velx_df, bounded.nn_vely_df, bounded.grid)[..., -1, :]
    assert top_cells.abs().max() < 1e-4
    torch.testing.assert_close(bounded.prediction.fields["velfacex"], bounded.nn_velx_df)


def test_each_sample_gets_its_own_velocity_boundaries(model):
    batch = random_batch(FIELD_NAMES, model.config_keys, batch_size=2)
    inflow = flashx_config()
    inflow["boundary"]["left"] = {"type": "inflow_ins"}
    batch = FlashXBatch(dict(batch.fields), [flashx_config(), inflow], batch.config_keys)
    with torch.no_grad():
        model.debed.linear.weight.normal_(std=0.3)
    output = _run(model, model.normalizer.normalize(batch), use_velocity_boundaries=True)
    assert output.nn_velx_df[0, ..., 0].abs().max() == 0
    # an inflow keeps the network's velocity
    assert output.nn_velx_df[1, ..., 0].abs().max() > 0
    assert output.nn_velx_df[1, ..., -1].abs().max() == 0


def test_an_unknown_boundary_type_is_rejected(model):
    config = flashx_config()
    config["boundary"]["top"] = {"type": "periodic"}
    batch = random_batch(FIELD_NAMES, model.config_keys, batch_size=1)
    batch = FlashXBatch(dict(batch.fields), [config], batch.config_keys)
    _run(model, model.normalizer.normalize(batch))  # only read when asked for
    with pytest.raises(ValueError, match="no face velocity condition for boundary type 'periodic'"):
        _run(model, model.normalizer.normalize(batch), use_velocity_boundaries=True)


def test_checkpoints_from_before_the_option_keep_the_temperature_ansatz():
    config = _model().get_extra_state()["config"]
    del config["use_temperature_ansatz"]
    assert Nucleus2MoEDivFreeProjection.config_from_dict(config).use_temperature_ansatz


def test_a_checkpoint_saved_under_the_old_model_name_loads(model, tmp_path):
    state = model.state_dict()
    state["_extra_state"] = {**state["_extra_state"], "model_name": "nucleus2_moe_divfree"}
    torch.save({"model": state}, tmp_path / "checkpoint.pt")
    loaded = load_model(tmp_path / "checkpoint.pt")
    assert isinstance(loaded, Nucleus2MoEDivFreeProjection)
    assert loaded.get_extra_state()["model_name"] == "nucleus2_moe_divfree_projection"
