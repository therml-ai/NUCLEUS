import random

import pytest
import torch
from hydra import compose, initialize

from nucleus.models.modules import get_train_module
from nucleus.models.nucleus2_moe_divfree import (
    GRADIENT_SPACING,
    Nucleus2MoEDivFree,
    Nucleus2MoEDivFreeConfig,
    Nucleus2MoEDivFreeInput,
    Nucleus2MoEDivFreeOutput,
    divfree_input_to_cells,
)
from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch, DivFreeData
from nucleus.trajectory import Trajectory
from nucleus.data.normalize import DivFreeNormalizer, NormalizerConstants
from nucleus.physics.poisson import divergence_centers_from_faces
from nucleus.utils.losses import sdf_sign_bce_loss


class _IdentityParamsNormalizer(DivFreeNormalizer):
    """Unit per-field stats make the field normalize/unnormalize identity, and the
    sim-parameter normalization is identity so ``forward_trajectory`` builds its
    conditioning tensor without needing min/max dicts."""

    def normalize_params(self, sim_params_dicts):
        return sim_params_dicts

    def unnormalize_params(self, sim_params_dicts):
        return sim_params_dicts


def make_normalizer() -> _IdentityParamsNormalizer:
    constants = NormalizerConstants(
        max_domain_size=9.0, sdf_mean=0.0, sdf_std=1.0, absmax_temp=1.0,
        temp_mean=0.0, temp_std=1.0, velx_mean=0.0, velx_std=1.0, vely_mean=0.0,
        vely_std=1.0, psi_mean=0.0, psi_std=1.0, phi_mean=0.0, phi_std=1.0,
    )
    return _IdentityParamsNormalizer(constants)


@pytest.fixture
def model() -> Nucleus2MoEDivFree:
    return Nucleus2MoEDivFree(Nucleus2MoEDivFreeConfig(
        patch_size=16, embed_dim=32, num_heads=2, processor_blocks=1,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
    )).eval()


def make_divfree_data(batch_size, time, height, width, sdf_fill=None) -> DivFreeData:
    """Random fields on their natural grids (with a batch dim). ``sdf_fill`` fills
    the sdf with a constant instead of noise (e.g. deep liquid)."""
    def rand(*trailing):
        return torch.randn(batch_size, time, *trailing)

    sdf = rand(height, width) if sdf_fill is None else torch.full(
        (batch_size, time, height, width), float(sdf_fill)
    )
    return DivFreeData(
        sdf=sdf, temperature=rand(height, width),
        velx=rand(height, width + 1), vely=rand(height + 1, width),
        psi=rand(height + 1, width + 1), phi=rand(height, width),
    )


def as_input(data: DivFreeData) -> Nucleus2MoEDivFreeInput:
    return Nucleus2MoEDivFreeInput(data.sdf, data.temperature, data.velx, data.vely)


def temp_sim_params_dict() -> dict:
    """The physical scalars step's temperature ansatz reads: bulk_temp (used to
    unnormalize the network's temperature output to physical units), the saturation
    temperature, and the heater's wall temperature and x-extent."""
    return {
        "bulk_temp": 50.0, "sat_temp": 58.0,
        "heater": {"wallTemp": 80.0, "xMin": 0.4, "xMax": 1.6},
    }


def x_coords_for(width: int) -> torch.Tensor:
    """Cell-center x positions, shape ``(width,)``, in the same units as the heater
    x-extent above."""
    return (torch.arange(width, dtype=torch.float32) + 0.5) * GRADIENT_SPACING


def make_batch(data: DivFreeData, sim_params_tensor: torch.Tensor) -> DivFreeBatch:
    batch_size = data.sdf.shape[0]
    return DivFreeBatch(
        input=data,
        target=None,
        sim_params=[temp_sim_params_dict() for _ in range(batch_size)],
        dx=torch.full((batch_size,), GRADIENT_SPACING),
        dy=torch.full((batch_size,), GRADIENT_SPACING),
        sim_params_tensor=sim_params_tensor,
    )


def run_step(model, data, sim_params, normalizer, input_type, use_div_gate=False, use_leray=False):
    """Feed the model input as either a bare ``Nucleus2MoEDivFreeInput`` (with the
    sim-param tensor and dict passed separately) or a ``DivFreeBatch`` (which carries
    them). Both paths supply the x-coordinates the temperature ansatz needs."""
    x_coords = x_coords_for(data.sdf.shape[-1])
    if input_type == "input":
        return model.step(
            as_input(data), sim_params, normalizer, x_coords=x_coords,
            sim_params_dict=[temp_sim_params_dict() for _ in range(data.sdf.shape[0])],
            use_div_gate=use_div_gate, use_leray=use_leray,
        )
    return model.step(
        make_batch(data, sim_params), normalizer=normalizer, x_coords=x_coords,
        use_div_gate=use_div_gate, use_leray=use_leray,
    )


def make_sim_params_dict(model) -> dict:
    # forward_trajectory assembles its conditioning tensor from a physical sim-param
    # dict; the values are arbitrary for a finiteness/shape check. expected_heater_params
    # already includes xMin/xMax, which the temperature ansatz's heater band also reads.
    params = {"bulk_temp": 50.0, "sat_temp": 58.0}
    params.update({param: random.random() for param in model.expected_fluid_params})
    params["heater"] = {param: random.random() for param in model.expected_heater_params}
    params.update({param: random.random() for param in model.expected_global_params})
    return params


@pytest.mark.parametrize("input_type", ["input", "batch"])
def test_step_returns_output_on_natural_grids(model, input_type):
    torch.manual_seed(0)
    batch_size, time, height, width = 1, 2, 64, 64
    data = make_divfree_data(batch_size, time, height, width)
    sim_params = torch.randn(batch_size, model.num_sim_params)

    with torch.no_grad():
        output = run_step(model, data, sim_params, make_normalizer(), input_type)

    assert isinstance(output, Nucleus2MoEDivFreeOutput)
    assert output.sdf.shape == (batch_size, time, height, width)
    assert output.temperature.shape == (batch_size, time, height, width)
    assert output.velx.shape == (batch_size, time, height, width + 1)
    assert output.vely.shape == (batch_size, time, height + 1, width)
    assert output.phi.shape == (batch_size, time, height, width)
    for field in (output.sdf, output.temperature, output.velx, output.vely, output.phi):
        assert torch.isfinite(field).all()


def test_temperature_ansatz_uses_normalized_sat_temp(model):
    # sim_params_dict carries sat_temp/heater wallTemp as physical values, but
    # temperature_ansatz adds them directly to nn (the network's normalized output:
    # field = sat_temp + interface_decay * nn). step() must normalize them first --
    # otherwise output.temperature would be dominated by the raw physical value
    # (tens of degrees) instead of staying in normalized units like every other field.
    torch.manual_seed(0)
    width = 64
    with torch.no_grad():
        # bias=False on the debed's Linear, so zeroing its only weight makes every
        # raw head output (nn_sdf, nn_temp, ...) exactly 0 everywhere, independent of
        # the input -- so interface_decay(sdf_physical=0) == 0 and nn == 0, and the
        # (sdf==0) field term collapses to exactly sat_temp_normalized regardless of
        # what the network would otherwise (arbitrarily, untrained) predict.
        model.debed.linear.weight.zero_()

    data = make_divfree_data(1, 2, 64, width)
    sim_params = torch.randn(1, model.num_sim_params)
    # A non-identity temp normalizer, so a physical-vs-normalized mixup actually
    # produces a detectably wrong value instead of accidentally matching.
    constants = NormalizerConstants(
        max_domain_size=9.0, sdf_mean=0.0, sdf_std=1.0, absmax_temp=1.0,
        temp_mean=10.0, temp_std=4.0, velx_mean=0.0, velx_std=1.0, vely_mean=0.0,
        vely_std=1.0, psi_mean=0.0, psi_std=1.0, phi_mean=0.0, phi_std=1.0,
    )
    normalizer = _IdentityParamsNormalizer(constants)
    sim_params_dict = {
        "bulk_temp": 50.0, "sat_temp": 58.0,
        "heater": {"wallTemp": 80.0, "xMin": 0.4, "xMax": 1.6},
    }

    with torch.no_grad():
        output = model.step(
            as_input(data), sim_params, normalizer, x_coords=x_coords_for(width),
            sim_params_dict=[sim_params_dict],
        )

    expected_sat_temp_normalized = ((58.0 - 50.0) - constants.temp_mean) / constants.temp_std
    # The last column (x far past heater xMax=1.6) is outside the heater band for
    # every row, so the heater gate is ~1 there and the result reduces to the field
    # value above.
    far_from_heater = output.temperature[..., -1]
    assert torch.allclose(
        far_from_heater, torch.full_like(far_from_heater, expected_sat_temp_normalized), atol=1e-4
    )


def test_step_with_sdf_reinit_gate(model):
    # Redistancing the sdf before the divergence gate is toggled by use_sdf_reinit;
    # the output stays on the natural grids and finite.
    torch.manual_seed(0)
    batch_size, time, height, width = 1, 2, 64, 64
    data = make_divfree_data(batch_size, time, height, width)
    sim_params = torch.randn(batch_size, model.num_sim_params)

    with torch.no_grad():
        output = model.step(
            as_input(data), sim_params, make_normalizer(),
            x_coords=x_coords_for(width), sim_params_dict=[temp_sim_params_dict()],
            use_sdf_reinit=True,
        )

    assert output.velx.shape == (batch_size, time, height, width + 1)
    for field in (output.sdf, output.velx, output.vely, output.phi):
        assert torch.isfinite(field).all()


def test_step_uses_per_sample_sim_params_dict(model):
    # Each batch element's temperature ansatz must use its own physical params --
    # not sample 0's dict applied to the whole batch. make_batch/make_divfree_batch
    # give every sample the identical dict, so this needs genuinely different dicts
    # per sample to actually exercise per-sample indexing.
    torch.manual_seed(0)
    height, width = 64, 64
    data0 = make_divfree_data(1, 2, height, width)
    data1 = make_divfree_data(1, 2, height, width)
    data = DivFreeData(
        sdf=torch.cat([data0.sdf, data1.sdf]),
        temperature=torch.cat([data0.temperature, data1.temperature]),
        velx=torch.cat([data0.velx, data1.velx]),
        vely=torch.cat([data0.vely, data1.vely]),
        psi=torch.cat([data0.psi, data1.psi]),
        phi=torch.cat([data0.phi, data1.phi]),
    )
    sim_params = torch.randn(2, model.num_sim_params)
    normalizer = make_normalizer()
    x_coords = x_coords_for(width)

    dict0 = temp_sim_params_dict()
    dict1 = {"bulk_temp": 35.0, "sat_temp": 40.0, "heater": {"wallTemp": 55.0, "xMin": -0.5, "xMax": 0.2}}

    with torch.no_grad():
        batched = model.step(
            as_input(data), sim_params, normalizer, x_coords=x_coords,
            sim_params_dict=[dict0, dict1],
        )
        solo0 = model.step(
            as_input(data0), sim_params[:1], normalizer, x_coords=x_coords,
            sim_params_dict=[dict0],
        )
        solo1 = model.step(
            as_input(data1), sim_params[1:], normalizer, x_coords=x_coords,
            sim_params_dict=[dict1],
        )

    assert torch.allclose(batched.temperature[0], solo0.temperature[0])
    assert torch.allclose(batched.temperature[1], solo1.temperature[0])
    # sanity: the two samples' physical params actually differ enough to matter.
    assert not torch.allclose(batched.temperature[0], batched.temperature[1])


@pytest.mark.parametrize("input_type", ["input", "batch"])
def test_leray_solenoidal_part_is_divergence_free(model, input_type):
    # With use_leray, the solenoidal velocity (velx_sol/vely_sol) is the Leray projection
    # of the raw predicted velocity, so it is divergence free everywhere regardless of the
    # sdf or the divergence source (which only enters velx_dil).
    torch.manual_seed(0)
    batch_size, time, height, width = 1, 2, 64, 64
    with torch.no_grad():
        model.debed.linear.weight.normal_()  # nonzero predicted velocity to project
    data = make_divfree_data(batch_size, time, height, width)
    sim_params = torch.randn(batch_size, model.num_sim_params)

    with torch.no_grad():
        output = run_step(model, data, sim_params, make_normalizer(), input_type, use_leray=True)

    # The unit normalizer keeps velx_sol/vely_sol physical, so their MAC divergence is the
    # Leray residual.
    divergence = divergence_centers_from_faces(
        output.velx_sol, output.vely_sol, GRADIENT_SPACING, GRADIENT_SPACING
    )
    interior = divergence[..., :-1, :]  # drop the top outflow row
    assert interior.abs().max() < 1e-2


def make_divfree_batch(batch_size, time, height, width, num_sim_params) -> DivFreeBatch:
    return DivFreeBatch(
        input=make_divfree_data(batch_size, time, height, width),
        target=make_divfree_data(batch_size, time, height, width),
        sim_params=[temp_sim_params_dict() for _ in range(batch_size)],
        dx=torch.full((batch_size,), GRADIENT_SPACING),
        dy=torch.full((batch_size,), GRADIENT_SPACING),
        sim_params_tensor=torch.randn(batch_size, num_sim_params),
    )


def build_divfree_module(extra_overrides=()):
    """Build a small CPU-sized DivFreeForecastModule from the real configs."""
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(
            config_name="default",
            overrides=[
                "model_cfg=nucleus2/nucleus2_divfree",
                "normalizer_cfg=divfree",
                "data_dir=/tmp",
                "model_cfg.train_module_name=divfree_forecast",
                "model_cfg.params.patch_size=16",
                "model_cfg.params.embed_dim=32",
                "model_cfg.params.processor_blocks=1",
                "model_cfg.params.num_heads=2",
                "model_cfg.params.num_experts=2",
                "model_cfg.params.topk=1",
                "model_cfg.params.moe_intermediate_dim=32",
                "model_cfg.params.activation_dtype=float32",
                *extra_overrides,
            ],
        )

    Module = get_train_module(cfg.model_cfg.train_module_name)
    assert Module.__name__ == "DivFreeForecastModule"
    module = Module(
        cfg.checkpoint_path,
        cfg.model_cfg,
        cfg.data_cfg,
        cfg.normalizer_cfg,
        cfg.optim_cfg,
        cfg.scheduler_cfg,
        log_wandb=False,
        normalization_constants=None,
    )
    module.default_log_dict = lambda *args, **kwargs: None  # force no logging
    return module


def test_divfree_forecast_module(model):
    # Build the DivFreeForecastModule around the Nucleus2MoEDivFree model via its
    # config, then run a validation step on a DivFreeBatch.
    module = build_divfree_module()
    assert isinstance(module.model, Nucleus2MoEDivFree)

    batch = make_divfree_batch(2, 2, 64, 64, module.model.num_sim_params)

    @torch.compiler.disable
    def check_step():
        with torch.no_grad():
            loss = module.validation_step(batch, 0)
            assert torch.isfinite(loss)
            assert loss > 0
    check_step()


def _zero_output():
    return Nucleus2MoEDivFreeOutput(
        **{field: torch.zeros(1, 1, 4, 4) for field in
           ("sdf", "temperature", "phi", "div_source", "gated_div_source")},
        velx=torch.zeros(1, 1, 4, 5), vely=torch.zeros(1, 1, 5, 4),
        velx_sol=torch.zeros(1, 1, 4, 5), vely_sol=torch.zeros(1, 1, 5, 4),
        velx_dil=torch.zeros(1, 1, 4, 5), vely_dil=torch.zeros(1, 1, 5, 4),
        moe_outputs=[],
    )


def _target(**overrides):
    base = dict(
        sdf=torch.zeros(1, 1, 4, 4), temperature=torch.zeros(1, 1, 4, 4),
        velx=torch.zeros(1, 1, 4, 5), vely=torch.zeros(1, 1, 5, 4),
        psi=torch.zeros(1, 1, 4, 4), phi=torch.zeros(1, 1, 4, 4),
    )
    base.update(overrides)
    return DivFreeData(**base)


def test_field_loss_is_mean_abs_error_over_supervised_grids():
    # _field_loss is the summed |error| over exactly the grids _loss_fields selects,
    # divided by their total element count. This stays correct regardless of which
    # field set _loss_fields currently supervises.
    module = build_divfree_module()
    output = _zero_output()
    target = _target(
        sdf=torch.ones(1, 1, 4, 4), temperature=torch.ones(1, 1, 4, 4),
        velx=torch.ones(1, 1, 4, 5), vely=torch.ones(1, 1, 5, 4),
        psi=torch.ones(1, 1, 4, 4), phi=torch.ones(1, 1, 4, 4),
    )

    pred_grids, target_grids = module._loss_fields(output, target)
    total_abs_error = sum(
        (pred - tgt).abs().sum() for pred, tgt in zip(pred_grids, target_grids)
    )
    total_elements = sum(pred.numel() for pred in pred_grids)
    expected = total_abs_error / total_elements

    assert torch.allclose(module._field_loss(output, target), expected)
    assert module._field_loss(output, target) > 0


def test_field_loss_ignores_unsupervised_fields():
    # A mismatch in a field that _loss_fields does not select leaves the loss at zero.
    module = build_divfree_module()
    output = _zero_output()

    supervised_fields = {
        name for name in ("sdf", "temperature", "velx", "vely", "phi")
        if any(
            grid is getattr(output, name)
            for grid in module._loss_fields(output, _target())[0]
        )
    }
    unsupervised = ({"sdf", "temperature", "velx", "vely", "phi"}
                    - supervised_fields)

    field_shape = {
        "sdf": (4, 4), "temperature": (4, 4), "velx": (4, 5),
        "vely": (5, 4), "phi": (4, 4),
    }
    for name in unsupervised:
        mismatch = _target(**{name: torch.ones(1, 1, *field_shape[name])})
        assert module._field_loss(output, mismatch) == 0, name


def test_augment_preserves_field_shapes():
    # Per-field augmentation runs each field through the module's augmentations on its
    # own natural grid, preserving shapes and staying finite.
    torch.manual_seed(0)
    module = build_divfree_module()
    data = make_divfree_data(2, 2, 64, 64)
    augmented = module._augment(data)
    for name in ("sdf", "temperature", "velx", "vely", "psi", "phi"):
        assert augmented.__getattribute__(name).shape == data.__getattribute__(name).shape
        assert torch.isfinite(augmented.__getattribute__(name)).all()


def test_per_field_mae_logs_every_field():
    module = build_divfree_module()
    output = _zero_output()
    target = DivFreeData(
        sdf=torch.ones(1, 1, 4, 4), temperature=torch.ones(1, 1, 4, 4),
        velx=torch.ones(1, 1, 4, 5), vely=torch.ones(1, 1, 5, 4),
        psi=torch.ones(1, 1, 4, 4), phi=torch.ones(1, 1, 4, 4),
    )
    metrics = module._per_field_mae(output, target, "train")
    # Every field is logged, plus the auxiliary velx_sol/vely_sol/velx_dil/vely_dil
    # and div_source targets.
    assert set(metrics) == {f"train/mae_{name}" for name in
                            ("sdf", "temperature", "velx", "vely",
                             "velx_sol", "vely_sol", "velx_dil", "vely_dil", "div_source")}
    for name in ("sdf", "temperature", "velx", "vely"):
        assert metrics[f"train/mae_{name}"] == pytest.approx(1.0)
    # The target velocity is spatially constant, so its divergence (the div_source
    # target) is zero, matching the zero prediction.
    assert metrics["train/mae_div_source"] == pytest.approx(0.0)


@pytest.mark.parametrize("trajectory_steps", [8, 24])
def test_forward_trajectory(model, trajectory_steps):
    torch.manual_seed(0)
    batch_size, height, width = 1, 64, 64
    input_window = 8
    # forward_trajectory takes and returns a Trajectory with every field on its
    # natural grid.
    data = make_divfree_data(batch_size, input_window, height, width)
    initial_state = Trajectory(
        sdf=data.sdf, temp=data.temperature, velx=data.velx, vely=data.vely,
        sim_params=[make_sim_params_dict(model)],
    )

    with torch.inference_mode():
        trajectory = model.forward_trajectory(
            initial_state,
            make_normalizer(),
            dx=GRADIENT_SPACING,
            input_time_window_size=input_window,
            output_time_window_size=input_window,
            trajectory_steps=trajectory_steps,
        )

    assert isinstance(trajectory, Trajectory)
    assert trajectory.num_steps == trajectory_steps
    for field in (trajectory.sdf, trajectory.temp, trajectory.velx, trajectory.vely):
        assert field.shape[0] == batch_size
        assert field.shape[1] == trajectory_steps
        assert field.isfinite().all()
    assert trajectory.sdf.shape[-2:] == (height, width)
    assert trajectory.velx.shape[-2:] == (height, width + 1)
    assert trajectory.vely.shape[-2:] == (height + 1, width)


def test_debed_diagnostics_cover_all_heads():
    # The per-head diagnostics should report a grad norm and a weight distribution for
    # each debed head. FlexAttention has no CPU backward, so populate the head gradients
    # directly -- _debed_diagnostics only reads .grad and the weights.
    module = build_divfree_module()
    for head in module._debed_heads().values():
        for param in head.parameters():
            param.grad = torch.randn_like(param)

    scalar_log, histogram_log = module._debed_diagnostics(
        include_scalars=True, include_histograms=True
    )
    for name in ("debed",):
        assert f"train/{name}/grad_norm" in scalar_log
        assert scalar_log[f"train/{name}/grad_norm"] > 0
        assert torch.isfinite(scalar_log[f"train/{name}/grad_norm"])
        assert f"train/{name}/weight_std" in scalar_log
        assert f"train/{name}/grad_hist" in histogram_log
        assert f"train/{name}/weight_hist" in histogram_log

    # With histograms disabled, only the scalar diagnostics are produced.
    scalars_only, empty_histograms = module._debed_diagnostics(
        include_scalars=True, include_histograms=False
    )
    assert empty_histograms == {}
    assert "train/debed/grad_norm" in scalars_only
    

def test_input_embedding_has_six_channels():
    # The embedding consumes sdf, temperature, and the four split face-velocity
    # channels -- psi/phi are not part of the input.
    assert Nucleus2MoEDivFree.expected_fields == [
        "dfun", "temperature", "vel_left", "vel_right", "vel_bottom", "vel_top",
    ]
    data = make_divfree_data(1, 2, 8, 8)
    cells = divfree_input_to_cells(data.sdf, data.temperature, data.velx, data.vely)
    assert cells.shape == (1, 2, 8, 8, 6)


def test_input_potentials_do_not_affect_output(model):
    # psi/phi are excluded from the input embedding, so perturbing them must leave the
    # model output unchanged.
    torch.manual_seed(0)
    data = make_divfree_data(1, 2, 64, 64)
    sim_params = torch.randn(1, model.num_sim_params)
    normalizer = make_normalizer()

    kwargs = dict(x_coords=x_coords_for(64), sim_params_dict=[temp_sim_params_dict()])
    with torch.no_grad():
        baseline = model.step(as_input(data), sim_params, normalizer, **kwargs)
        perturbed = DivFreeData(
            sdf=data.sdf, temperature=data.temperature, velx=data.velx, vely=data.vely,
            psi=torch.randn_like(data.psi), phi=torch.randn_like(data.phi),
        )
        perturbed_output = model.step(as_input(perturbed), sim_params, normalizer, **kwargs)

    for name in ("sdf", "temperature", "velx", "vely", "phi"):
        assert torch.equal(getattr(baseline, name), getattr(perturbed_output, name)), name


def test_div_gate_confines_divergence_to_band(model):
    # Option B: the model predicts a divergence source; the Poisson reconstruction makes
    # div(velocity) = gated source. So with use_div_gate=True the velocity divergence is
    # ~0 wherever the (predicted-sdf) band gate is zero -- the divergence itself is
    # confined, and grad(phi) still extends smoothly into the bulk. The gate off (the
    # training default) leaves divergence leaking outside the band.
    torch.manual_seed(0)
    with torch.no_grad():
        # A nonzero debed makes the predicted sdf span an interface and the divergence
        # source (the 5th debed channel) nonzero.
        model.debed.linear.weight.normal_(std=0.3)

    data = make_divfree_data(1, 2, 64, 64)
    sim_params = torch.randn(1, model.num_sim_params)
    normalizer = make_normalizer()

    # use_leray makes the solenoidal part divergence-free, so the total velocity's
    # divergence equals the (gated) source -- otherwise the raw predicted velocity leaks
    # its own divergence everywhere and the confinement can't be measured.
    kwargs = dict(x_coords=x_coords_for(64), sim_params_dict=[temp_sim_params_dict()], use_leray=True)
    with torch.no_grad():
        gated = model.step(as_input(data), sim_params, normalizer, use_div_gate=True, **kwargs)
        ungated = model.step(as_input(data), sim_params, normalizer, use_div_gate=False, **kwargs)

    # The band gate is built from the predicted (physical) sdf the model gated on.
    sdf_physical = normalizer.unnormalize_sdf(gated.sdf)
    in_band = sdf_physical.abs() <= 5 * GRADIENT_SPACING
    assert in_band.any() and (~in_band).any()  # both regions exist for the test to mean anything

    def divergence(output):
        return divergence_centers_from_faces(
            output.velx, output.vely, GRADIENT_SPACING, GRADIENT_SPACING
        )

    # Drop the top outflow row where the Dirichlet BC breaks the exact div = source identity.
    interior = (slice(None), slice(None), slice(None, -1), slice(None))
    gated_div = divergence(gated)[interior]
    outside = ~in_band[interior]
    inside = in_band[interior]

    assert gated_div[outside].abs().max() < 1e-2  # divergence confined: ~0 outside the band
    assert gated_div[inside].abs().max() > 1e-1   # the source injects real divergence in-band

    # Ungated, the same source leaks divergence outside the band.
    assert divergence(ungated)[interior][outside].abs().max() > 1e-1


def test_div_source_supervised_by_target_velocity_divergence():
    # div_source enters the loss with a target computed from the (normalized) target
    # velocities: divergence_centers_from_faces(target.velx, target.vely).
    torch.manual_seed(0)
    module = build_divfree_module()
    output = _zero_output()  # div_source predicted as 0
    target = _target(velx=torch.randn(1, 1, 4, 5), vely=torch.randn(1, 1, 5, 4))

    pred_grids, target_grids = module._loss_fields(output, target)

    # The divergence source is the last (pred, target) pair in the loss.
    assert pred_grids[-1] is output.div_source
    expected_target = divergence_centers_from_faces(
        target.velx, target.vely, GRADIENT_SPACING, GRADIENT_SPACING
    )
    assert torch.allclose(target_grids[-1], expected_target)
    # The target velocity is not divergence-free, so the div_source term contributes.
    assert (pred_grids[-1] - target_grids[-1]).abs().sum() > 0


def test_sdf_sign_loss_handles_all_liquid_batch():
    # The real config's sdf_mean is negative, so an all-zero normalized sdf
    # unnormalizes to physical sdf < 0 everywhere (all liquid) -- no vapor pixels,
    # so the num_liquid/num_vapor ratio is undefined and must be guarded rather
    # than dividing by zero.
    module = build_divfree_module()
    loss = module._sdf_sign_loss(_zero_output(), _target())
    assert torch.isfinite(loss)


def test_sdf_sign_loss_weights_by_target_liquid_vapor_ratio():
    module = build_divfree_module()
    output = _zero_output()
    sdf = torch.zeros(1, 1, 4, 4)
    sdf[0, 0, 0, :3] = 5.0  # 3 cells normalize to a large-positive (vapor) physical sdf
    target = _target(sdf=sdf)

    sdf_physical = module.normalizer.unnormalize_sdf(sdf)
    num_vapor = (sdf_physical > 0).sum()
    num_liquid = sdf_physical.numel() - num_vapor
    expected_weight = (num_liquid / num_vapor).item()
    pred_sdf_physical = module.normalizer.unnormalize_sdf(output.sdf)
    expected_loss = sdf_sign_bce_loss(pred_sdf_physical, sdf_physical, expected_weight)

    assert torch.allclose(module._sdf_sign_loss(output, target), expected_loss)


def test_step_use_mass_transfer_replaces_learned_source(model):
    # use_mass_transfer swaps the learned divergence-source channel for the physics
    # continuity source (mdot * n.grad(rho)); it needs the physical sim-param dict.
    torch.manual_seed(0)
    data = make_divfree_data(1, 2, 64, 64)
    sim_params = torch.randn(1, model.num_sim_params)
    normalizer = make_normalizer()
    sim_params_dict = make_sim_params_dict(model)
    sim_params_dict.update(
        {"stefan": 0.1, "inv_reynolds": 0.01, "prandtl": 1.0, "thcogas": 0.1, "rhogas": 0.01}
    )
    # above bulk_temp so temp non-dim is sane; xMin/xMax feed the temperature ansatz.
    sim_params_dict["heater"].update({"wallTemp": 80.0, "xMin": 0.4, "xMax": 1.6})
    x_coords = x_coords_for(64)

    with torch.no_grad():
        learned = model.step(
            as_input(data), sim_params, normalizer,
            x_coords=x_coords, sim_params_dict=[sim_params_dict],
        )
        physics = model.step(
            as_input(data), sim_params, normalizer, x_coords=x_coords,
            use_mass_transfer=True, sim_params_dict=[sim_params_dict],
        )

    assert physics.div_source.shape == learned.div_source.shape
    for field in (physics.sdf, physics.velx, physics.vely, physics.div_source):
        assert torch.isfinite(field).all()
    # The physics source is not the learned head output.
    assert not torch.allclose(physics.div_source, learned.div_source)

    # Without x_coords / the sim-param dict, step can't build the temperature ansatz.
    with torch.no_grad(), pytest.raises(AssertionError):
        model.step(as_input(data), sim_params, normalizer, use_mass_transfer=True)
