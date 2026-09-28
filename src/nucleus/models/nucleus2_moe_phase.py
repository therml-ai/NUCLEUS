import dataclasses
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.profiler import record_function
from rotary_embedding_torch import RotaryEmbedding
from typing import Literal

from nucleus.layers.adaptive_layernorm import AdaptiveLayerNorm
from nucleus.layers.attention import NeighborhoodAttention
from nucleus.layers.moe.topk_moe import TopkMoE, TopkMoEOutput, TopkRouterWithBias
from nucleus.layers.droppath import DropPath
from nucleus.layers import (
    AdaptiveEmbed,
    AdaptiveDebed,
    LinearEmbed,
    LinearDebed
)
from nucleus.data.batching import CollatedBatch
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from nucleus.models.nucleus2_moe import Nucleus2MoE, Nucleus2MoEConfig, get_dtype

from ._api import register_model

__all__ = ["Nucleus2MoEPhase", "Nucleus2MoEConfig"]

@register_model("nucleus2_moe_phase")
@torch.compile(fullgraph=True)
class Nucleus2MoEPhase(Nucleus2MoE):
    expected_fields = ["temperature", "velx", "vely"]

    def __init__(self, config: Nucleus2MoEConfig):
        super().__init__(config)
        self.config = config
        self.phase_embed = nn.Embedding(num_embeddings=2, embedding_dim=self.config.embed_dim)
        self.phase_patcher = LinearEmbed(
            patch_size=config.patch_size,
            in_channels=config.embed_dim,
            embed_dim=config.embed_dim,
            dtype=self.embed_dtype
        )
        self.phase_unpatcher = LinearDebed(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            out_channels=1,
            dtype=self.debed_dtype
        )

    def forward(self, batch: CollatedBatch) -> torch.Tensor:
        phase = (batch.input[..., 0] > 0).to(torch.int32)
        fields = batch.input[..., 1:]
        return self.step(fields, phase, batch.sim_params_tensor)
    
    def _soft_phase_embed(self, phase_logits: torch.Tensor) -> torch.Tensor:
        # Blend the liquid/vapor embedding rows by the predicted vapor probability
        # instead of looking up a hard index. This keeps the field head's gradient
        # flowing back into the phase head, so the heads are trained to agree rather
        # than just sharing a trunk.
        vapor_prob = torch.sigmoid(phase_logits).unsqueeze(-1)   # (B, T, H, W, 1)
        liquid_embed = self.phase_embed.weight[0]                # (D,)
        vapor_embed = self.phase_embed.weight[1]                 # (D,)
        return liquid_embed + vapor_prob * (vapor_embed - liquid_embed)

    def step(self, input: torch.Tensor, phase: torch.Tensor, sim_params: torch.Tensor):
        r"""
        Args:
            input: [B, T, H, W, <temp, velx, vely>] a tensor with three channels
            phase: [B, T, H, W] int32 binary mask. 0 - liquid, 1 - vapor.
            sim_params: global simulation parameters
        """
        assert input.dtype == torch.float32
        assert phase.dtype == torch.int32
        assert sim_params.dtype == torch.float32
        
        _, _, h, w, _ = input.shape

        with record_function("field embed"):
            field_patches = self.embed(input.to(self.embed_dtype))
            
        with record_function("patch embed"):
            pe = self.phase_embed(phase)
            phase_patches = self.phase_patcher(pe)

        field_patches = field_patches.to(get_dtype(self.config.activation_dtype))
        phase_patches = phase_patches.to(get_dtype(self.config.activation_dtype))
        x = field_patches + phase_patches

        with record_function("get_axial_freqs"):
            with torch.no_grad():
                _, embed_t, embed_h, embed_w, _ = x.shape
                rotary_freqs = self.rotary_emb.get_axial_freqs(embed_t, embed_h, embed_w)[None, :, :, :, None, :]

        moe_outputs = []
        for idx, blk in enumerate(self.blocks):
            with record_function(f"block_{idx}"):
                x, moe_output = blk(x, rotary_freqs, sim_params)
                moe_outputs.append(moe_output)
        
        with record_function("debed"):
            x = self.out_norm(
                x.to(self.debed_dtype) + 
                field_patches.to(self.debed_dtype) + 
                phase_patches.to(self.debed_dtype)
            ) 
            
            phase_logits = self.phase_unpatcher(x, target_shape=(h, w)).squeeze(-1)
            #soft_phase = self._soft_phase_embed(phase_logits).to(self.debed_dtype)
            #print(x.size(), soft_phase.size())
            fields = self.debed(x, target_shape=(h, w))
            
        #temp, velx, vely = fields.unbind(dim=-1)
        # TODO: clip temp based on saturation temp (need to pass to step...)
        # TODO: predict stream function???

        return fields, phase_logits, moe_outputs

    def forward_trajectory(
        self,
        initial_state: torch.Tensor,
        sim_params: torch.Tensor,
        dx: float,
        input_time_window_size: int,
        output_time_window_size: int,
        trajectory_steps: int,
        use_sdf_reinit: bool = False,
        return_moe_outputs: bool = False
    ):
        assert initial_state.dim() == 5, "initial state must be [B, T, H, W, C]"
        assert sim_params.dim() == 2, "fluid params must be [B, num_params]"
        assert initial_state.shape[0] == sim_params.shape[0]
        assert input_time_window_size == initial_state.shape[1]

        trajectory_with_sdf = initial_state.clone()
        sdf = trajectory_with_sdf[..., 0]
        phase = (sdf > 0).to(torch.int32)
        fields = trajectory_with_sdf[..., 1:]
        
        trajectory_moe_outputs = [] if return_moe_outputs else None

        for _ in range(input_time_window_size, trajectory_steps, output_time_window_size):
            input_fields = fields[:, -input_time_window_size:]
            input_phase = phase[:, -input_time_window_size:]
            pred_fields, pred_phase_logits, moe_outputs = self.step(input_fields, input_phase, sim_params)
            
            pred_phase = (pred_phase_logits > 0).to(torch.int32)        
    
            output_fields = pred_fields[:, -output_time_window_size:]
            output_phase = pred_phase[:, -output_time_window_size:]

            fields = torch.cat((fields, output_fields), dim=1)
            phase = torch.cat((phase, output_phase), dim=1)
            if return_moe_outputs:
                trajectory_moe_outputs.append(moe_outputs)

        trajectory = torch.cat((
            phase[..., None].to(torch.float32),
            fields
        ), dim=-1)

        if return_moe_outputs:
            return trajectory, trajectory_moe_outputs
        return trajectory