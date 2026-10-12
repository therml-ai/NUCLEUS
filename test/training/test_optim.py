import pytest
import torch
from torch import nn

from nucleus.training.optim import build_optimization


def _model():
    return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


@pytest.mark.parametrize("name", ["adamw", "adam", "muon"])
def test_each_optimizer_steps(name):
    model = _model()
    optimization = build_optimization(
        model, {"name": name, "params": {"lr": 1e-2, "weight_decay": 0.0}},
        {"name": "cosine", "params": {"eta_min": 0.0}}, max_steps=10,
    )
    before = model[0].weight.detach().clone()
    # weighted, because a LayerNorm's outputs always sum to the same value, which
    # would leave the linear layer without a gradient
    (model(torch.randn(3, 4)) * torch.randn(3, 4)).sum().backward()
    optimization.step()
    assert not torch.equal(before, model[0].weight)


def test_adamw_only_decays_matrices():
    groups = build_optimization(
        _model(), {"name": "adamw", "params": {"lr": 1e-3, "weight_decay": 0.1}},
        {"name": "cosine", "params": {"eta_min": 0.0}}, max_steps=10,
    ).optimizers[0].param_groups
    assert [len(group["params"]) for group in groups] == [1, 3]
    assert [group["weight_decay"] for group in groups] == [0.1, 0.0]


def _trapezoid_rates(warmup, cooldown, max_steps=10):
    """The learning rate of every step of a trapezoid peaking at 1."""
    optimization = build_optimization(
        _model(), {"name": "adam", "params": {"lr": 1.0}},
        {"name": "trapezoidal", "params": {"warmup": warmup, "cooldown": cooldown, "scale_factor": 0.1}},
        max_steps=max_steps,
    )
    rates = []
    for _ in range(max_steps):
        rates.append(optimization.learning_rate())
        optimization.step()
    return rates


def test_trapezoidal_fractions_are_of_the_whole_run():
    rates = _trapezoid_rates(warmup=0.2, cooldown=0.2)
    assert rates[0] == pytest.approx(0.1)
    assert rates[5] == pytest.approx(1.0)
    assert rates[-1] < 1.0


def test_trapezoid_without_a_cooldown_holds_the_peak_to_the_end():
    rates = _trapezoid_rates(warmup=4, cooldown=0)
    assert rates[0] == pytest.approx(0.1)
    assert rates[4:] == pytest.approx([1.0] * 6)


def test_trapezoid_without_a_warmup_starts_at_the_peak():
    rates = _trapezoid_rates(warmup=0, cooldown=4)
    assert rates[:6] == pytest.approx([1.0] * 6)
    assert rates[-1] < 1.0


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="Optimizer sgd"):
        build_optimization(_model(), {"name": "sgd", "params": {}}, {"name": "cosine", "params": {}}, 1)
