from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from lion_pytorch import Lion
from torch import nn
from torch.optim import Adam, AdamW, Muon, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

from nucleus.utils.lr_schedulers import CosineWarmupLR, TrapezoidalLR


@dataclass
class Optimization:
    """The optimizers of a model and the learning rate schedulers stepping them,
    advanced together once per optimizer step."""
    optimizers: List[Optimizer]
    schedulers: List[LRScheduler]

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in self.schedulers:
            scheduler.step()

    def learning_rate(self) -> float:
        return self.optimizers[0].param_groups[0]["lr"]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for optimizer, optimizer_state in zip(self.optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(optimizer_state)
        for scheduler, scheduler_state in zip(self.schedulers, state["schedulers"], strict=True):
            scheduler.load_state_dict(scheduler_state)


def build_optimization(model: nn.Module, optim_cfg: dict, scheduler_cfg: dict, max_steps: int) -> Optimization:
    """Build after the model is on its device: fused AdamW needs CUDA parameters."""
    optimizers = build_optimizers(model, optim_cfg["name"], dict(optim_cfg["params"]))
    schedulers = [
        build_scheduler(optimizer, scheduler_cfg["name"], scheduler_cfg["params"], max_steps)
        for optimizer in optimizers
    ]
    return Optimization(optimizers, schedulers)


def build_optimizers(model: nn.Module, name: str, params: dict) -> List[Optimizer]:
    fused = next(model.parameters()).is_cuda
    if name == "adamw":
        weight_decay = params.pop("weight_decay", 0.0)
        return [AdamW(weight_decay_param_groups(model, weight_decay), **params, fused=fused)]
    if name == "adam":
        return [Adam(model.parameters(), **params)]
    if name == "lion":
        return [Lion(model.parameters(), **params)]
    if name == "muon":
        # Muon orthogonalizes matrix updates, so it only takes the 2d weights
        matrices = [param for param in model.parameters() if param.dim() == 2]
        others = [param for param in model.parameters() if param.dim() != 2]
        return [
            AdamW(others, **params, fused=fused),
            Muon(matrices, **params, adjust_lr_fn="match_rms_adamw"),
        ]
    raise ValueError(f"Optimizer {name} not supported")


def weight_decay_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    # Only weight-decay >=2D weights (matmul / conv). 1D params -- biases,
    # LayerNorm/RMSNorm gains, scalar scales -- are excluded so weight decay
    # can't shrink them toward zero.
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer: Optimizer, name: str, params: dict, max_steps: int) -> LRScheduler:
    if name == "cosine_warmup":
        return CosineWarmupLR(
            optimizer, warmup_iters=params["warmup"], max_iters=max_steps, eta_min=params["eta_min"]
        )
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=params["eta_min"])
    if name == "trapezoidal":
        warmup = _iterations(params["warmup"], max_steps)
        cooldown = _iterations(params["cooldown"], max_steps)
        return TrapezoidalLR(
            optimizer,
            scale_factor=params["scale_factor"],
            warmup_iters=warmup,
            flat_iters=max_steps - warmup - cooldown,
            cooldown_iters=cooldown,
        )
    raise ValueError(f"Scheduler {name} not supported")


def _iterations(value: float, max_steps: int) -> int:
    # a float is a fraction of the run, an int a number of steps
    return int(value * max_steps) if isinstance(value, float) else int(value)
