import torch
import torch.nn as nn
from nucleus.layers.moe.topk_moe import TopkMoE

def count_model_parameters(module: nn.Module, active: bool = False) -> int:
    r"""
    utility to count the number of parameters in a model.
    Args:
        module (nn.Module): the module to count the parameters of.
        active (bool): If True, only count the MoE active parameters. Otherwise,
                       count the parameters of every expert.
    Returns:
        int: the number of parameters in the model.
    """
    if isinstance(module, TopkMoE) and active:
        expert_active_params = module.topk * (module.w1[0].numel() + module.w2[0].numel())
        router_params = sum(p.numel() for p in module.router.parameters())
        return expert_active_params + router_params
    else:
        mod_params = module.parameters(recurse=False)
        child_params = [count_model_parameters(child, active) for child in module.children()]
        return sum(m.numel() for m in mod_params) + sum(child_params)