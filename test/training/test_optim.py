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


def test_trapezoidal_fractions_are_of_the_whole_run():
    optimization = build_optimization(
        _model(), {"name": "adam", "params": {"lr": 1.0}},
        {"name": "trapezoidal", "params": {"warmup": 0.2, "cooldown": 0.2, "scale_factor": 0.1}}, max_steps=10,
    )
    rates = []
    for _ in range(10):
        rates.append(optimization.learning_rate())
        optimization.step()
    assert rates[0] == pytest.approx(0.1)
    assert rates[5] == pytest.approx(1.0)
    assert rates[-1] < 1.0


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="Optimizer sgd"):
        build_optimization(_model(), {"name": "sgd", "params": {}}, {"name": "cosine", "params": {}}, 1)
