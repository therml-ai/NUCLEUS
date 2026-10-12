from typing import Dict, List

import torch
from torch import nn

from nucleus.layers.moe.topk_moe import TopkMoEOutput, TopkRouterWithBias


def router_loss(moe_outputs: List[TopkMoEOutput], load_balance_weight: float, z_loss_weight: float) -> torch.Tensor:
    """The auxiliary loss of routers trained by a loss; zero for routers balanced
    by a bias, which update_router_bias balances instead."""
    if moe_outputs[0].router_output.router_type() != "loss":
        return torch.zeros((), device=moe_outputs[0].router_output.z_loss.device)
    load_balance = sum(output.router_output.load_balance_loss for output in moe_outputs)
    z_loss = sum(output.router_output.z_loss for output in moe_outputs)
    return load_balance * load_balance_weight + z_loss * z_loss_weight


def update_router_bias(model: nn.Module, moe_outputs: List[TopkMoEOutput]) -> None:
    if moe_outputs[0].router_output.router_type() != "bias":
        return
    routers = [module for module in model.modules() if isinstance(module, TopkRouterWithBias)]
    for router, output in zip(routers, moe_outputs, strict=True):
        router.update_router_bias(output.router_output.tokens_per_expert)


def moe_metrics(moe_outputs: List[TopkMoEOutput], prefix: str) -> Dict[str, torch.Tensor]:
    metrics = {}
    for layer, output in enumerate(moe_outputs):
        tokens_per_expert = output.router_output.tokens_per_expert.float()
        router_logits = output.router_output.router_logits
        mean_tokens = tokens_per_expert.mean()
        metrics |= {
            f"{prefix}_moe/mean_router_logit_layer{layer}": router_logits.mean(),
            f"{prefix}_moe/max_router_logit_layer{layer}": router_logits.abs().max(),
            f"{prefix}_moe/coeff_of_variation_layer{layer}": tokens_per_expert.std() / mean_tokens,
            f"{prefix}_moe/load_imbalance_factor_layer{layer}": tokens_per_expert.max() / mean_tokens,
            # an expert routed under 1% of the tokens is counted as idle
            f"{prefix}_moe/active_experts_layer{layer}": (
                tokens_per_expert > tokens_per_expert.sum() * 0.01
            ).float().mean(),
        }
    return metrics
