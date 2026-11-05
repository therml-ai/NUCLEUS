import ast
import pathlib
import random

import pytest
import torch

from nucleus.models import get_model
from nucleus.models.modules import get_train_module
from nucleus.models.nucleus2_moe_staggered_backbone import (
    Nucleus2MoEStaggeredBackbone,
    StaggeredFields,
    faces_from_cell_split,
    staggered_to_cells,
)
from nucleus.models.nucleus2_moe_staggered import (
    Nucleus2MoEStaggeredBase,
    Nucleus2MoEStaggeredConfig,
    Nucleus2MoEStaggeredOutput,
    StaggeredForecastModule,
)
from nucleus.data.in_mem_divfree_forecast_dataset import DivFreeBatch, DivFreeData
from nucleus.data.normalize import DivFreeNormalizer, NormalizerConstants
from nucleus.trajectory import Trajectory
from hydra import compose, initialize


def small_config() -> Nucleus2MoEStaggeredConfig:
    return Nucleus2MoEStaggeredConfig(
        patch_size=16, embed_dim=32, num_heads=2, processor_blocks=1,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
    )


@pytest.fixture
def model() -> Nucleus2MoEStaggeredBase:
    return Nucleus2MoEStaggeredBase(small_config()).eval()


class _IdentityParamsNormalizer(DivFreeNormalizer):
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


def make_input(batch_size, time, height, width) -> StaggeredFields:
    return StaggeredFields(
        sdf=torch.randn(batch_size, time, height, width),
        temperature=torch.randn(batch_size, time, height, width),
        velx=torch.randn(batch_size, time, height, width + 1),
        vely=torch.randn(batch_size, time, height + 1, width),
    )


def test_faces_from_cell_split_inverts_input_split():
    # Splitting true faces onto cells and averaging back must return them exactly:
    # both copies of an interior face hold the same value.
    torch.manual_seed(0)
    height, width = 7, 5
    sdf, temp = torch.randn(2, 3, height, width), torch.randn(2, 3, height, width)
    velx, vely = torch.randn(2, 3, height, width + 1), torch.randn(2, 3, height + 1, width)
    cells = staggered_to_cells(sdf, temp, velx, vely)
    velx_back, vely_back = faces_from_cell_split(*cells[..., 2:].unbind(-1))
    assert torch.equal(velx_back, velx)
    assert torch.equal(vely_back, vely)


def test_faces_from_cell_split_averages_interior_and_keeps_walls():
    height, width = 3, 4
    left = torch.full((height, width), 1.0)
    right = torch.full((height, width), 3.0)
    bottom = torch.full((height, width), -2.0)
    top = torch.full((height, width), 4.0)
    velx, vely = faces_from_cell_split(left, right, bottom, top)
    assert velx.shape == (height, width + 1) and vely.shape == (height + 1, width)
    assert torch.all(velx[:, 0] == 1.0) and torch.all(velx[:, -1] == 3.0)
    assert torch.all(velx[:, 1:-1] == 2.0)
    assert torch.all(vely[0] == -2.0) and torch.all(vely[-1] == 4.0)
    assert torch.all(vely[1:-1] == 1.0)


def test_step_returns_faces(model):
    torch.manual_seed(0)
    batch_size, time, height, width = 2, 2, 64, 48
    sim_params = torch.randn(batch_size, model.num_sim_params)
    with torch.no_grad():
        output = model.step(make_input(batch_size, time, height, width), sim_params)
    assert isinstance(output, Nucleus2MoEStaggeredOutput)
    assert output.sdf.shape == (batch_size, time, height, width)
    assert output.temperature.shape == (batch_size, time, height, width)
    assert output.velx.shape == (batch_size, time, height, width + 1)
    assert output.vely.shape == (batch_size, time, height + 1, width)
    for field in (output.sdf, output.temperature, output.velx, output.vely):
        assert torch.isfinite(field).all()
    assert output.to_cell_tensor().shape == (batch_size, time, height, width, 6)


def test_step_accepts_batch_and_ignores_physics_kwargs(model):
    # The training loop passes x_coords / sim_params_dict for the physics model; the
    # data-only model accepts and ignores them, and a DivFreeBatch carries sim_params.
    torch.manual_seed(0)
    # patch 16 needs >= 3 tokens per side for the 3x3 neighborhood attention, so >= 48 cells
    data = make_input(1, 2, 64, 64)
    sim_params = torch.randn(1, model.num_sim_params)
    with torch.no_grad():
        direct = model.step(data, sim_params, make_normalizer(),
                            x_coords=torch.zeros(64), sim_params_dict=[{}])
        batch = DivFreeBatch(input=data, target=None, sim_params=[{}],
                             dx=torch.ones(1), dy=torch.ones(1), sim_params_tensor=sim_params)
        via_batch = model.step(batch)
    assert torch.allclose(direct.velx, via_batch.velx)


def test_registered_model_and_train_module():
    # The registered model is the compiled class; check it by behavior, not isinstance.
    model = get_model("nucleus2_moe_staggered", **vars(small_config()))
    assert getattr(model, "staggered_io", False)
    assert len(model.expected_fields) == 6
    assert get_train_module("staggered_forecast") is StaggeredForecastModule


def test_forward_trajectory_extends_on_natural_grids(model):
    torch.manual_seed(0)
    height, width, window = 64, 64, 4
    params = {"bulk_temp": 50.0, "sat_temp": 58.0}
    params.update({p: random.random() for p in model.expected_fluid_params if p not in params})
    params["heater"] = {p: random.random() for p in model.expected_heater_params}
    params.update({p: random.random() for p in model.expected_global_params})
    trajectory = Trajectory(
        sdf=torch.randn(1, window, height, width),
        temp=torch.randn(1, window, height, width),
        velx=torch.randn(1, window, height, width + 1),
        vely=torch.randn(1, window, height + 1, width),
        sim_params=[params],
    )
    with torch.no_grad():
        out = model.forward_trajectory(
            trajectory, make_normalizer(), dx=1 / 32,
            input_time_window_size=window, output_time_window_size=window, trajectory_steps=12,
        )
    assert out.num_steps == 12
    assert out.velx.shape[-2:] == (height, width + 1)
    assert out.vely.shape[-2:] == (height + 1, width)
    assert torch.isfinite(out.velx).all() and torch.isfinite(out.vely).all()


MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "nucleus" / "models"


def _imported_modules(path: pathlib.Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("filename", ["nucleus2_moe_staggered.py", "nucleus2_moe_staggered_backbone.py"])
def test_does_not_depend_on_divfree(filename):
    # The data-only model and the shared backbone must not import the physics model:
    # a change there should not be able to change this model or its checkpoints.
    # (The divfree *dataset* module is fine: it holds the shared data containers.)
    imported = _imported_modules(MODELS_DIR / filename)
    assert not any(name.endswith("nucleus2_moe_divfree") for name in imported), imported


def test_backbone_parameters_match_divfree():
    # The divfree model subclasses the backbone with a 5-channel head. Its parameters
    # must keep the backbone's names and shapes, so checkpoints saved before it did still
    # load. Extra parameters it adds later are allowed.
    config = small_config()
    backbone = Nucleus2MoEStaggeredBackbone(config, out_channels=5)
    divfree = get_model("nucleus2_moe_divfree", **vars(config))
    backbone_shapes = {k: v.shape for k, v in backbone.state_dict().items() if k != "_extra_state"}
    divfree_shapes = {k: v.shape for k, v in divfree.state_dict().items() if k != "_extra_state"}
    assert backbone_shapes.items() <= divfree_shapes.items()


def test_checkpoint_round_trip(tmp_path):
    from nucleus.models import load_model_from_checkpoint
    torch.manual_seed(0)
    model = get_model("nucleus2_moe_staggered", **vars(small_config()))
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    loaded = load_model_from_checkpoint(str(path))
    # Loading turns dtype strings into torch.dtype; compare the serialized form.
    from nucleus.models.nucleus2_moe_staggered_backbone import config_to_dict
    assert config_to_dict(loaded.config) == config_to_dict(model.config)
    for key, value in model.state_dict().items():
        if key != "_extra_state":
            assert torch.equal(value, loaded.state_dict()[key]), key


def build_staggered_module(extra_overrides=()):
    """Small CPU-sized StaggeredForecastModule from the real configs."""
    with initialize(version_base=None, config_path="../../config"):
        cfg = compose(
            config_name="default",
            overrides=[
                "model_cfg=nucleus2/nucleus2_staggered",
                "normalizer_cfg=divfree",
                "data_dir=/tmp",
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
    assert Module is StaggeredForecastModule
    module = Module(
        cfg.checkpoint_path, cfg.model_cfg, cfg.data_cfg, cfg.normalizer_cfg,
        cfg.optim_cfg, cfg.scheduler_cfg, log_wandb=False, normalization_constants=None,
    )
    module.default_log_dict = lambda *args, **kwargs: None
    module.get_current_lr = lambda: 0.0  # no trainer attached
    return module


def make_data(batch_size, time, height, width) -> DivFreeData:
    return DivFreeData(
        sdf=torch.randn(batch_size, time, height, width),
        temperature=torch.randn(batch_size, time, height, width),
        velx=torch.randn(batch_size, time, height, width + 1),
        vely=torch.randn(batch_size, time, height + 1, width),
        psi=torch.randn(batch_size, time, height + 1, width + 1),
        phi=torch.randn(batch_size, time, height, width),
    )


def make_batch(batch_size, time, height, width, num_sim_params) -> DivFreeBatch:
    return DivFreeBatch(
        input=make_data(batch_size, time, height, width),
        target=make_data(batch_size, time, height, width),
        sim_params=[{} for _ in range(batch_size)],
        dx=torch.full((batch_size,), 1 / 32),
        dy=torch.full((batch_size,), 1 / 32),
        sim_params_tensor=torch.randn(batch_size, num_sim_params),
    )


def test_field_loss_is_mae_over_the_four_fields():
    module = build_staggered_module()
    height, width = 4, 4
    output = Nucleus2MoEStaggeredOutput(
        sdf=torch.zeros(1, 1, height, width), temperature=torch.zeros(1, 1, height, width),
        velx=torch.zeros(1, 1, height, width + 1), vely=torch.zeros(1, 1, height + 1, width),
        moe_outputs=[],
    )
    target = DivFreeData(
        sdf=torch.full((1, 1, height, width), 1.0), temperature=torch.full((1, 1, height, width), 2.0),
        velx=torch.full((1, 1, height, width + 1), 3.0), vely=torch.full((1, 1, height + 1, width), 4.0),
        # psi and phi are not supervised: a mismatch there must not change the loss.
        psi=torch.full((1, 1, height + 1, width + 1), 100.0), phi=torch.full((1, 1, height, width), 100.0),
    )
    n_cell, n_x, n_y = height * width, height * (width + 1), (height + 1) * width
    expected = (1 * n_cell + 2 * n_cell + 3 * n_x + 4 * n_y) / (2 * n_cell + n_x + n_y)
    assert torch.isclose(module._field_loss(output, target), torch.tensor(expected))


def test_training_loss_is_data_plus_router_only():
    # The training objective is the L1 data loss plus the router terms, nothing else.
    torch.manual_seed(0)
    module = build_staggered_module(["model_cfg.params.augmentation=noise"])  # no field dropout
    module.noise.skip_prob = 1.0  # no input noise (the noise is drawn but multiplied by 0)
    module.model.eval()  # no drop-path, so the two forward passes agree
    batch = make_batch(2, 2, 64, 64, module.model.num_sim_params)

    @torch.compiler.disable
    def run():
        with torch.no_grad():
            output = module.model.step(StaggeredForecastModule._model_input(batch.input), batch.sim_params_tensor)
            aux_loss, _ = module._router_loss(output.moe_outputs)
            expected = module._field_loss(output, batch.target) + aux_loss
            loss = module.training_step(batch, 0)
        assert torch.allclose(loss, expected)
    run()


def test_validation_step_runs():
    torch.manual_seed(0)
    module = build_staggered_module()
    batch = make_batch(2, 2, 64, 64, module.model.num_sim_params)

    @torch.compiler.disable
    def run():
        with torch.no_grad():
            loss = module.validation_step(batch, 0)
        assert torch.isfinite(loss) and loss > 0
    run()


def test_unknown_augmentation_is_rejected():
    with pytest.raises(ValueError):
        build_staggered_module(["model_cfg.params.augmentation=flip"])


def test_cell_centered_augmentation_drops_whole_fields():
    module = build_staggered_module(["model_cfg.params.augmentation=cell_centered"])
    module.noise.skip_prob = 1.0           # no noise
    module.field_dropout.active_prob = 1.0
    module.field_dropout.p = 1.0           # every field dropped
    data = make_data(3, 2, 8, 6)
    out = module._augment(data)
    for name in ("sdf", "temperature", "velx", "vely"):
        assert torch.all(getattr(out, name) == 0), name
    # psi/phi are not model inputs and are left alone.
    assert torch.equal(out.psi, data.psi) and torch.equal(out.phi, data.phi)


def test_cell_centered_noise_scale_is_shared_across_fields():
    torch.manual_seed(0)
    module = build_staggered_module(["model_cfg.params.augmentation=cell_centered"])
    module.noise.skip_prob = 0.0
    module.field_dropout.active_prob = 0.0  # no dropout
    batch_size = 4
    data = make_data(batch_size, 2, 96, 96)
    out = module._augment(data)
    stds = torch.stack([
        (getattr(out, name) - getattr(data, name)).flatten(1).std(dim=1)
        for name in ("sdf", "temperature", "velx", "vely")
    ])  # (fields, batch)
    # Within a sample every field gets the same noise scale ...
    assert torch.allclose(stds, stds[0:1].expand_as(stds), rtol=0.05)
    # ... and the scale is drawn per sample (log-uniform over [1e-3, 5e-2]).
    assert stds[0].max() / stds[0].min() > 1.5


def test_staggered_standard_normalizer_matches_standard():
    from omegaconf import OmegaConf
    from nucleus.data.normalize import StandardNormalizer, get_normalizer
    cfg_dir = pathlib.Path(__file__).resolve().parents[2] / "config" / "normalizer_cfg"
    load = lambda name: OmegaConf.to_container(OmegaConf.load(cfg_dir / f"{name}.yaml"), resolve=True)
    staggered, standard = load("staggered_standard"), load("standard")
    assert isinstance(get_normalizer(staggered), StandardNormalizer)
    for key, value in standard.items():
        assert staggered[key] == value, key
    for key in ("psi_mean", "psi_std", "phi_mean", "phi_std"):
        assert staggered[key] is not None
