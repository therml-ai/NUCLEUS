import dataclasses
import io
import pytest
import torch

from nucleus.models import (
    get_model,
    load_model_from_checkpoint,
    Nucleus1ViTConfig,
    Nucleus1MoEConfig,
    Nucleus2MoEConfig,
)

NUCLEUS1_VIT_MODELS = ["nucleus1_vit", "nucleus1_axial_vit", "nucleus1_neighbor_vit"]
NUCLEUS1_MOE_MODELS = ["nucleus1_vit_moe", "nucleus1_axial_moe", "nucleus1_moe"]
NUCLEUS2_MOE_MODELS = ["nucleus2_moe"]

NUCLEUS1_VIT_KWARGS = dict(
    input_fields=4, output_fields=4, patch_size=4, embed_dim=64,
    num_heads=4, processor_blocks=2, mlp_ratio=4.0,
)
NUCLEUS1_MOE_KWARGS = dict(
    input_fields=4, output_fields=4, patch_size=4, embed_dim=64,
    num_heads=4, processor_blocks=2, num_experts=4, topk=2, mlp_ratio=4.0,
)
NUCLEUS2_MOE_KWARGS = dict(
    patch_size=4, embed_dim=64, num_heads=4, processor_blocks=2,
    num_experts=4, topk=2, moe_intermediate_dim=256,
    patching="Linear"
)

EXPECTED_CONFIGS = {
    **{name: (NUCLEUS1_VIT_KWARGS, Nucleus1ViTConfig) for name in NUCLEUS1_VIT_MODELS},
    **{name: (NUCLEUS1_MOE_KWARGS, Nucleus1MoEConfig) for name in NUCLEUS1_MOE_MODELS},
    **{name: (NUCLEUS2_MOE_KWARGS, Nucleus2MoEConfig) for name in NUCLEUS2_MOE_MODELS},
}


def _save_raw(model) -> io.BytesIO:
    """Simulate torch.save(model.state_dict(), path)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return buf


def _save_lightning(model) -> io.BytesIO:
    """Simulate a Lightning checkpoint: state_dict nested under 'state_dict' with 'model.' prefix."""
    prefixed = {f"model.{k}": v for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save({"state_dict": prefixed}, buf)
    buf.seek(0)
    return buf


@pytest.mark.parametrize("model_name", list(EXPECTED_CONFIGS))
def test_config_attached_after_construction(model_name):
    kwargs, expected_cls = EXPECTED_CONFIGS[model_name]
    model = get_model(model_name, **kwargs)
    assert hasattr(model, "config"), "model missing .config attribute"
    assert isinstance(model.config, expected_cls)


@pytest.mark.parametrize("model_name", list(EXPECTED_CONFIGS))
def test_model_name_stored(model_name):
    kwargs, _ = EXPECTED_CONFIGS[model_name]
    model = get_model(model_name, **kwargs)
    assert model._model_name == model_name


@pytest.mark.parametrize("model_name", list(EXPECTED_CONFIGS))
def test_config_in_state_dict(model_name):
    kwargs, _ = EXPECTED_CONFIGS[model_name]
    model = get_model(model_name, **kwargs)
    sd = model.state_dict()
    assert "_extra_state" in sd
    assert sd["_extra_state"]["model_name"] == model_name
    expected = model.get_extra_state()["config"]
    assert sd["_extra_state"]["config"] == expected


@pytest.mark.parametrize("model_name", list(EXPECTED_CONFIGS))
def test_load_from_raw_checkpoint(model_name):
    """torch.save(model.state_dict(), path) -> load_model_from_checkpoint."""
    kwargs, _ = EXPECTED_CONFIGS[model_name]
    model = get_model(model_name, **kwargs)
    buf = _save_raw(model)
    loaded = load_model_from_checkpoint(buf)
    assert loaded._model_name == model_name
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert torch.equal(p1, p2), f"weight mismatch at {n1}"


@pytest.mark.parametrize("model_name", list(EXPECTED_CONFIGS))
def test_load_from_lightning_checkpoint(model_name):
    """Lightning-format checkpoint -> load_model_from_checkpoint."""
    kwargs, _ = EXPECTED_CONFIGS[model_name]
    model = get_model(model_name, **kwargs)
    buf = _save_lightning(model)
    loaded = load_model_from_checkpoint(buf)
    assert loaded._model_name == model_name
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
        assert torch.equal(p1, p2), f"weight mismatch at {n1}"
