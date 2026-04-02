import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

def load_balance_loss(
    router_logits: torch.Tensor,
    expert_counts: torch.Tensor,
    topk: int,
    num_experts: int
):
    r"""
    Switch-transformer load balance loss.
    This computes the dot product of two vectors:
        1. the percentage of tokens routed to each expert.
        2. the average routing probability of each expert.
    Args:
        router_logits: (num_tokens, num_experts)
        expert_counts: (num_experts,)
        num_tokens: int
    """
    mean_tokens_per_expert = expert_counts.to(torch.float32) / expert_counts.to(torch.float32).sum()
    router_probs = F.softmax(router_logits, dim=-1)
    mean_router_prob_per_expert = router_probs.mean(dim=0)
    loss = torch.dot(mean_tokens_per_expert, mean_router_prob_per_expert) * num_experts
    return loss

def z_loss(router_logits: torch.Tensor):
    return (torch.logsumexp(router_logits, dim=-1) ** 2).mean()

def get_token_indices(
    topk_expert_indices: torch.Tensor, 
    num_experts: int
):
    # The routing choices are not used in the backward pass, so this can always use no_grad.
    with torch.no_grad():
        flat_expert_indices = topk_expert_indices.view(-1)
        # The argsort returns int64 indices, but we don't need more than 32 bits for the indices.
        indices = flat_expert_indices.argsort().to(torch.int32)
        tokens_per_expert = torch.histc(flat_expert_indices, min=0, max=num_experts - 1, bins=num_experts)
        # group indices initialized like this to use int32.
        group_indices = torch.empty(tokens_per_expert.size(0), dtype=torch.int32, device=topk_expert_indices.device)
        torch.cumsum(tokens_per_expert, dim=0, out=group_indices)
        return group_indices, indices, tokens_per_expert
    
@dataclass
class RouterOutput:
    router_logits: torch.Tensor
    topk_probs: torch.Tensor
    topk_indices: torch.Tensor
    group_indices: torch.Tensor
    indices: torch.Tensor
    tokens_per_expert: torch.Tensor
    load_balance_loss: torch.Tensor
    z_loss: torch.Tensor
    
    def router_type(self):
        return None
    
    def to(self, device: torch.device):
        return RouterOutput(
            router_logits=self.router_logits.to(device),
            topk_probs=self.topk_probs.to(device),
            topk_indices=self.topk_indices.to(device),
            group_indices=self.group_indices.to(device),
            indices=self.indices.to(device),
            tokens_per_expert=self.tokens_per_expert.to(device),
            load_balance_loss=self.load_balance_loss.to(device),
            z_loss=self.z_loss.to(device),
        )
    
    def detach(self):
        return RouterOutput(
            router_logits=self.router_logits.detach(),
            topk_probs=self.topk_probs.detach(),
            topk_indices=self.topk_indices.detach(),
            group_indices=self.group_indices.detach(),
            indices=self.indices.detach(),
            tokens_per_expert=self.tokens_per_expert.detach(),
            load_balance_loss=self.load_balance_loss.detach(),
            z_loss=self.z_loss.detach(),
        )
    
class RouterBase(nn.Module):
    r"""
    This is a base class for a typical Top-k MoE router: Topk(Softmax(Rx))
    """
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        topk: int,
        softmax_first: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.softmax_first = softmax_first
        # Need bias=False, since a bias would create preference for specific experts,
        # regardless of the input data.
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        # scale is chosen to be smaller than typical default initalization,
        # smaller inital parameters should result in better initial load balance.
        with torch.no_grad():
            self.router.weight.data.normal_(0, 0.01)
        
    def forward(self, x, router_bias: Optional[torch.Tensor] = None):       
        router_logits = self.router(x)

        score = torch.clamp(router_logits, min=-2, max=2)
        
        if router_bias is not None:
            biased_score = score + router_bias[None, :]
            _, topk_indices = torch.topk(biased_score, k=self.topk, dim=-1)
            topk_scores = torch.gather(score, dim=-1, index=topk_indices)
        else:
            topk_scores, topk_indices = torch.topk(score, k=self.topk, dim=-1)
        
        topk_probs = F.softmax(topk_scores, dim=-1)
                
        group_indices, indices, tokens_per_expert = get_token_indices(topk_indices, self.num_experts)
        
        return RouterOutput(
            router_logits=router_logits,
            topk_probs=topk_probs,
            topk_indices=topk_indices,
            group_indices=group_indices,
            indices=indices,
            tokens_per_expert=tokens_per_expert,
            load_balance_loss=load_balance_loss(
                router_logits, 
                tokens_per_expert, 
                self.topk, 
                self.num_experts
            ),
            z_loss=z_loss(router_logits),
        )

@dataclass
class TopkRouterWithLossOutput(RouterOutput):    
    def router_type(self):
        return "loss"
    
class TopkRouterWithLoss(RouterBase):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        topk: int,
        softmax_first: bool,
    ):
        super().__init__(num_experts, hidden_dim, topk, softmax_first)

    def forward(self, x):
        router_output = super().forward(x)
        return TopkRouterWithLossOutput(
            router_logits=router_output.router_logits,
            topk_probs=router_output.topk_probs,
            topk_indices=router_output.topk_indices,
            group_indices=router_output.group_indices,
            indices=router_output.indices,
            tokens_per_expert=router_output.tokens_per_expert,
            load_balance_loss=router_output.load_balance_loss,
            z_loss=router_output.z_loss,
        )
        
@dataclass
class TopkRouterWithBiasOutput(RouterOutput):
    router_bias: torch.Tensor
    
    def router_type(self):
        return "bias"
        
class TopkRouterWithBias(RouterBase):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        topk: int,
        bias_update_rate: float,
        softmax_first: bool,
    ):
        super().__init__(num_experts, hidden_dim, topk, softmax_first)
        # The router bias is adjusted based on load balance, not modified during the backward pass.
        self.register_buffer("router_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.router_bias_update_rate = bias_update_rate
        self.target_load_ratio = 1 / num_experts # each expert should get an equal ratio of tokens
    
    @torch.no_grad()
    def update_router_bias(self, tokens_per_expert: torch.Tensor):
        assert tokens_per_expert.dim() == 1, "Expert counts must be of shape (num_experts,)"
        total_tokens = tokens_per_expert.sum().float()
        load_ratio = tokens_per_expert / total_tokens
        increase_mask = load_ratio < self.target_load_ratio
        decrease_mask = ~increase_mask
        self.router_bias[increase_mask] += self.router_bias_update_rate
        self.router_bias[decrease_mask] -= self.router_bias_update_rate
    
    def forward(self, x):
        router_output = super().forward(x, self.router_bias)
        return TopkRouterWithBiasOutput(
            router_logits=router_output.router_logits,
            topk_probs=router_output.topk_probs,
            topk_indices=router_output.topk_indices,
            group_indices=router_output.group_indices,
            indices=router_output.indices,
            tokens_per_expert=router_output.tokens_per_expert,
            router_bias=self.router_bias,
            load_balance_loss=router_output.load_balance_loss,
            z_loss=router_output.z_loss,
        )
        
@dataclass
class TopkMoEOutput:
    out: torch.Tensor
    router_output: RouterOutput
    topk: int
    num_experts: int
    
    def to(self, device: torch.device):
        return TopkMoEOutput(
            out=self.out.to(device),
            router_output=self.router_output.to(device),
            topk=self.topk,
            num_experts=self.num_experts,
        )
    
    def detach(self):
        return TopkMoEOutput(
            out=self.out.detach(),
            router_output=self.router_output.detach(),
            topk=self.topk,
            num_experts=self.num_experts,
        )

# grouped_mm is cannot be cuda-graphed due to a host-device transfer, so cannot use "reduced-overhead"
# (The offsets `offs` are moved to the CPU, and then moved back to the device when iterating over each gemm.)
@torch.compile(fullgraph=True)
class TopkMoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        topk: int,
        router: RouterBase,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.topk = topk
        
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_dim, hidden_dim, dtype=torch.bfloat16))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16))
        self.reset_parameters()

        self.router = router
        
    def reset_parameters(self):
        with torch.no_grad():
            gain = math.sqrt(2)
            self.w1.data.normal_(0, gain / math.sqrt(self.hidden_dim))
            self.w2.data.normal_(0, gain / math.sqrt(self.intermediate_dim))

    def forward(self, x):
        r"""
        This assumes that the input tensor is of shape (B, T, H, W, C).
        This is done so we can track the routing statistics for each patch and time step.
        """
        assert x.dim() == 5, "Input tensor must be of shape (B, T, H, W, C)"
        B, T, H, W, C = x.shape
        
        # flatten x because the MoE is applied to each patch independently.
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        batch_size = x.shape[0]
        
        router_output = self.router(x)
        
        # NOTE with torch.compile(fullgraph=True), the grouped gemm kernel does not support torch.float32, 
        # so the input data has to be truncated to bfloat 16.
        groups = x.to(torch.bfloat16)[router_output.indices // self.topk]        

        groups = torch.nn.functional.grouped_mm(groups, self.w1.mT, offs=router_output.group_indices)
        groups = F.gelu(groups)
        groups = torch.nn.functional.grouped_mm(groups, self.w2.mT, offs=router_output.group_indices)
        
        # Scatter the tokens to [B, topk, hidden_dim]
        scattered = torch.empty_like(groups)
        scattered[router_output.indices] = groups
        scattered = scattered.view(batch_size, self.topk, self.hidden_dim)
        
        # reduce the output tokens and scale by the routing probability
        out = (scattered * router_output.topk_probs.unsqueeze(-1)).sum(dim=1).view(input_shape)
        
        # Convert to convenient shape for visualization.
        router_output.topk_indices = router_output.topk_indices.view(B, T, H, W, self.topk).detach()
        
        return TopkMoEOutput(
            out=out.to(torch.float32), 
            router_output=router_output,
            topk=self.topk,
            num_experts=self.num_experts,
        )