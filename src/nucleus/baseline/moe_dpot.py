"""
The pytorch moduels are taken and modified from https://github.com/haiyangxin/MoEPOT,
commit sha c36ec22aa492b9460d2e9f24cf4f0354d0f07540. The only changes to the pytorch
modules are comments and some formatting improvements.

The configs and lightning module are custom and setup to be similar to the other model's training setup.

General Notes:
1. this uses the class token to train the model to recognize different input types. (This is perhaps
   not technically sufficient for different PDEs and liquids, since input/solution spaces for different 
   overlap. I.e., different flavors of linear PDEs may technically be different, but all valid for any functions
   in (say) L_1. I guess typically surrogate datasets are just chosen to have different flavors of
   initial conditions based on what's physically "interesting" and the input distributions won't really overlap
   for different PDEs.)
2. Routing is done on batch items, NOT patches. Different boiling setups (should) be routed to different experts.
3. The routing uses a surprisingly complex network and keeps CNN bias=True, which could be a problem if the bias
   does not naturally go to 0 during training.
"""

from datetime import date
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _WeightedLoss
from typing import Tuple
import numpy as np
from einops import rearrange
import lightning as L
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint

from nucleus.data.batching import CollatedBatch, collate
from nucleus.data.in_mem_forecast_dataset import InMemForecastDataset
from nucleus.data.normalize import get_normalizer
from nucleus.utils.parameter_count import count_model_parameters

ACTIVATION = {"gelu": nn.GELU()}

class SimpleLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, return_comps=False):
        super(SimpleLpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.return_comps = return_comps

    def forward(self, x, y, mask=None):
        num_examples = x.size()[0]
        # Lp loss 2
        if mask is not None:##TODO: will be meaned by n_channels for single channel data
            x = x * mask
            y = y * mask

            ## compute effective channels
            # msk_channels = mask.sum(dim=(1,2,3),keepdim=False).count_nonzero(dim=-1) # B, 1
            msk_channels = mask.sum(dim=list(range(1, mask.ndim-1)),keepdim=False).count_nonzero(dim=-1) # B, 1
        else:
            msk_channels = x.shape[-1]

        diff_norms = torch.norm(x.reshape(num_examples,-1, x.shape[-1]) - y.reshape(num_examples,-1,x.shape[-1]), self.p, dim=1)    ##N, C
        y_norms = torch.norm(y.reshape(num_examples,-1, y.shape[-1]), self.p, dim=1) + 1e-8
        if self.reduction:
            if self.size_average:
                    return torch.mean(diff_norms/y_norms)          ## deprecated
            else:
                return torch.sum(torch.sum(diff_norms/y_norms, dim=-1) / msk_channels)    #### go this branch
        else:
            return torch.sum(diff_norms/y_norms, dim=-1) / msk_channels

class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)

class GlobalTopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, initial_temperature=2.0, is_finetune=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = initial_temperature
        self.min_temperature = 0.5
        self.temperature_decay = 0.99
        
        if is_finetune:
            self.temperature = 0.5
            
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Note (Arthur): The gate does NOT set bias=False. So there may be some
        # going to be some inherent preference for certain experts,
        # unless the bias naturally goes to 0 during training.
        self.gate = nn.Sequential(
            nn.Conv2d(input_dim, input_dim*2, 1),
            nn.BatchNorm2d(input_dim*2),
            nn.GELU(),
            
            ChannelAttention(input_dim*2),
            
            nn.Conv2d(input_dim*2, input_dim, 1),
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, num_experts, 1)
        )
    
    def update_temperature(self):
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.temperature_decay
        )
    
    def forward(self, x):
        # NOTE: Routing is applied to batch items, not patches.
        global_feat = self.global_pool(x)  # [B, C, 1, 1]
        gating_scores = self.gate(global_feat).squeeze(-1).squeeze(-1)  # [B, num_experts]
        
        top_k_values, top_k_indices = torch.topk(gating_scores, self.top_k, dim=1)  # [B, top_k]
        top_k_values = F.softmax(top_k_values / self.temperature, dim=1)
        
        return top_k_indices, top_k_values

class Expert(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Expert, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class MoEImage(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, 
                 num_experts, shared_experts_num=2, top_k=4 ,is_finetune=False):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_experts = num_experts
        self.shared_experts_num = shared_experts_num
        self.top_k = top_k
        self.is_finetune = is_finetune

        self.feature_extractor = ConvFeatureExtractor(input_channels, hidden_channels)
        self.gating = GlobalTopKGating(hidden_channels, num_experts, top_k, is_finetune=self.is_finetune)
        
        self.shared_experts = nn.ModuleList([
            Expert(hidden_channels, output_channels) 
            for _ in range(shared_experts_num)
        ])
        
        self.experts = nn.ModuleList([
            Expert(hidden_channels, output_channels) 
            for _ in range(num_experts)
        ])

    def freeze_feature_and_gating(self, freeze=True):
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze
        for param in self.gating.parameters():
            param.requires_grad = not freeze
            
    def forward(self, x):
        features = self.feature_extractor(x)
        
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(features) / self.shared_experts_num
        
        output = torch.zeros_like(x)
        
        top_k_indices, top_k_values = self.gating(features)
        
        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx)
            weights = top_k_values * mask
            expert_output = self.experts[expert_idx](features)
            output += expert_output * weights.sum(dim=1).view(-1, 1, 1, 1)
        
        if self.training and not self.is_finetune:
            loss_gate = self.compute_balance_loss(top_k_values, top_k_indices)
            self.gating.update_temperature()
        else:
            loss_gate = 0

        return shared_output + output, loss_gate

    def compute_balance_loss(self, gates, indices):
        importance = torch.zeros(self.num_experts, device=gates.device)
        for i in range(self.num_experts):
            mask = (indices == i)
            importance[i] = (gates * mask).sum()
        
        ideal_load = gates.sum() / self.num_experts
        balance_loss = torch.pow(importance - ideal_load, 2).mean()
            
        return balance_loss

class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    """
    def __init__(
        self, 
        width=32, 
        num_blocks=8, 
        channel_first=False,
        sparsity_threshold=0.01, 
        modes=32,
        hard_thresholding_fraction=1, 
        hidden_size_factor=1, 
        act='gelu'
    ):
        super().__init__()
        assert width % num_blocks == 0, f"hidden_size {width} should be divisble by num_blocks {num_blocks}"
        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.hidden_size_factor = hidden_size_factor
        # self.scale = 0.02
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        self.act = act

        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

    ### N, C, X, Y
    def forward(self, x, spatial_size=None):
        if self.channel_first:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  ### ->N, X, Y, C
        else:
            B, H, W, C = x.shape
        x_orig = x

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        # total_modes = H*W // 2 + 1
        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes] = self.act(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        
        x = x + x_orig
        if self.channel_first:
            x = x.permute(0, 3, 1, 2)     ### N, C, X, Y

        return x


class Block(nn.Module):
    def __init__(
            self, 
            mixing_type='afno', 
            double_skip=True, 
            width=32,
            n_blocks=4, 
            mlp_ratio=1., 
            channel_first=True, 
            modes=32, 
            drop=0.,
            drop_path=0., 
            act='gelu',
            h=14, 
            w=8,
            is_finetune=False
        ):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = act

        if mixing_type == "afno":
            self.filter = AFNO2D(width = width, num_blocks=n_blocks, sparsity_threshold=0.01, channel_first=channel_first, modes=modes,
                                 hard_thresholding_fraction=1, hidden_size_factor=1, act=act)

        self.norm2 = torch.nn.GroupNorm(8, width)

        mlp_hidden_dim = int(width * mlp_ratio)
        self.MoE = MoEImage(width, mlp_hidden_dim, output_channels=width, num_experts=16, shared_experts_num=2, top_k=4, is_finetune=is_finetune)
        if is_finetune: # Freeze Gate Control Network
            self.MoE.freeze_feature_and_gating()
            self.MoE.feature_extractor.eval()
            self.MoE.gating.eval()
        self.double_skip = double_skip
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x, loss_gate = self.MoE(x)
        
        x = x + residual

        return x, loss_gate

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128, act='gelu'):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.out_dim = out_dim
        self.act = ACTIVATION[act]

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv2d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x
    
class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type='mlp'):
        super(TimeAggregator, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == 'mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels), requires_grad=True)   # initialization could be tuned
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels), requires_grad=True)   # initialization could be tuned
            self.gamma = nn.Parameter(2**torch.linspace(-10,10, out_channels).unsqueeze(0),requires_grad=True)  # 1, C
    
    # B, X, Y, T, C
    def forward(self, x):
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device) # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)

        return x

class MoEPOTNet(L.LightningModule):
    def __init__(
            self, 
            config: dict,
            router_loss_weight: float,
            lr: float
        ):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        self.in_channels = self.config["in_channels"]
        self.out_channels = self.config["out_channels"]
        self.in_timesteps = self.config["in_timesteps"]
        self.out_timesteps = self.config["out_timesteps"]
        self.n_blocks = self.config["n_blocks"]
        self.modes = self.config["modes"]
        self.num_features = self.embed_dim = embed_dim= self.config["embed_dim"]  # num_features for consistency with other models
        self.mlp_ratio = self.config["mlp_ratio"]
        self.act = ACTIVATION[self.config["act"]]
        self.out_layer_dim = self.config["out_layer_dim"]
        img_size = self.config["img_size"]
        patch_size = self.config["patch_size"]
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=self.in_channels + 3,
            embed_dim=self.out_channels * patch_size + 3,
            out_dim=embed_dim,
            act=self.config["act"],
        )
        self.latent_size = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1]))
        self.normalize = self.config["normalize"]
        self.time_agg = self.config["time_agg"]
        self.n_cls = self.config["n_cls"]
        self.is_finetune = self.config["is_finetune"]
    
        h = img_size // patch_size
        w = h // 2 + 1

        self.blocks = nn.ModuleList([
            Block(
                mixing_type=self.config["mixing_type"], 
                modes=self.modes,
                width=self.embed_dim, 
                mlp_ratio=self.mlp_ratio, 
                channel_first=True, 
                n_blocks=self.n_blocks, 
                double_skip=False,
                h=h, 
                w=w,
                act=self.act,
                is_finetune=self.is_finetune)
            for i in range(self.config["depth"])])

        if self.normalize:
            self.scale_feats_mu = nn.Linear(2 * self.in_channels, self.embed_dim)
            self.scale_feats_sigma = nn.Linear(2 * self.in_channels, self.embed_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, self.n_cls)
        )

        self.time_agg_layer = TimeAggregator(self.in_channels, self.in_timesteps, self.embed_dim, self.time_agg)
        
        ### attempt load balancing for high resolution
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.embed_dim, out_channels=self.out_layer_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv2d(in_channels=self.out_layer_dim, out_channels=self.out_layer_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv2d(in_channels=self.out_layer_dim, out_channels=self.out_channels * self.out_timesteps,kernel_size=1, stride=1)
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.mixing_type = self.config["mixing_type"]
        
        # Lightning Module settings
        self.router_loss_weight = router_loss_weight
        self.lr = lr
        self.data_loss = SimpleLpLoss(size_average=False)
        self.cls_loss = nn.CrossEntropyLoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)  # .02
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid

    def get_grid_3d(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, 1])

        grid = torch.cat((gridx, gridy, gridz), dim=-1)
        return grid

    ### in/out: B, X, Y, T, C
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, _, T, _ = x.shape # [8,128,128,10,1]
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3),keepdim=True), x.std(dim=(1,2,3),keepdim=True) + 1e-6    # B,1,1,1,C
            x = (x - mu)/ sigma
            scale_mu = self.scale_feats_mu(torch.cat([mu, sigma],dim=-1)).squeeze(-2).permute(0,3,1,2)   #-> B, C, 1, 1
            scale_sigma = self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1)).squeeze(-2).permute(0, 3, 1, 2)

        grid = self.get_grid_3d(x) 
        x = torch.cat((x, grid), dim=-1).contiguous() # B, X, Y, T, C+3
        x = rearrange(x, 'b x y t c -> (b t) c x y')
        x = self.patch_embed(x)
        x = x + self.pos_embed
         
        x = rearrange(x, '(b t) c x y -> b x y t c', b=B, t=T)

        x = self.time_agg_layer(x)

        x = rearrange(x, 'b x y c -> b c x y')

        if self.normalize:
            x = scale_sigma * x + scale_mu   ### Ada_in layer 
        
        loss_total = 0
        for blk in self.blocks:
            x, loss = blk(x)
            loss_total += loss

        cls_token = x.mean(dim=(2, 3), keepdim=False)
        cls_pred = self.cls_head(cls_token)

        x = self.out_layer(x).permute(0, 2, 3, 1)
        x = x.reshape(*x.shape[:3], self.out_timesteps, self.out_channels).contiguous()

        if self.normalize:
            x = x * sigma  + mu

        return x, cls_pred, loss_total

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' \
                              + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(
                    p[1].requires_grad) + ')\n'

        return string_repr
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def _cls_label(self, batch: CollatedBatch):
        r"""
        Constructs class labels with shape [B, len(table)].
        This is used for the CrossEntropyLoss.
        """
        liquids = ["fc72", "r515b", "ln2"]
        setups = ["subcooled", "saturated"]
        table = [f"{liquid}_{setup}" for setup in setups for liquid in liquids]
        labels = []
        labels_one_hot = []
        for i, d in enumerate(batch.fluid_params_dict):
            setup = d["setup"]
            liquid = d["liquid"]
            key = f"{liquid}_{setup}"
            index = table.index(key)
            one_hot = torch.zeros(len(table), device=batch.input.device, dtype=torch.float)
            one_hot[index] = 1
            labels.append(index)
            labels_one_hot.append(one_hot)
        return (
            torch.tensor(labels, device=batch.input.device, dtype=torch.long), 
            torch.stack(labels_one_hot, dim=0)
        )
        
    def training_step(self, batch: CollatedBatch, batch_idx: int):
        cls_indices_target, cls_one_hot_target = self._cls_label(batch)
        
        pred, cls_pred, router_loss_total = self.forward(batch.input)
        data_loss = self.data_loss(pred, batch.target)
        cls_loss = self.cls_loss(cls_pred, cls_indices_target)
        loss = data_loss + cls_loss + router_loss_total * self.router_loss_weight
        
        cls_pred_label = torch.argmax(cls_pred, dim=1)
        cls_pred_correct = (cls_pred_label == cls_indices_target).float().sum() / cls_pred_label.numel()
        
        mae_loss = torch.nn.functional.l1_loss(pred, batch.target)
        mse_loss = torch.nn.functional.mse_loss(pred, batch.target)
        absmax_error = (pred - batch.target).abs().max()
        
        self.log("train/data_loss", data_loss)
        self.log("train/cls_loss", cls_loss)
        self.log("train/unweighted_router_loss", router_loss_total)
        self.log("train/loss", loss)
        self.log("train/cls_pred_correct", cls_pred_correct)
        self.log("train/mae_loss", mae_loss)
        self.log("train/mse_loss", mse_loss)
        self.log("train/absmax_error", absmax_error)
        return loss
    
    def validation_step(self, batch: CollatedBatch, batch_idx: int):
        cls_indices_target, cls_one_hot_target = self._cls_label(batch)
        
        pred, cls_pred, router_loss_total = self.forward(batch.input)
        data_loss = self.data_loss(pred, batch.target)
        cls_loss = self.cls_loss(cls_pred, cls_indices_target)
        loss = data_loss + cls_loss + router_loss_total * self.router_loss_weight
        
        cls_pred_label = torch.argmax(cls_pred, dim=1)
        cls_pred_correct = (cls_pred_label == cls_indices_target).float().sum() / cls_pred_label.numel()
        
        mae_loss = torch.nn.functional.l1_loss(pred, batch.target)
        mse_loss = torch.nn.functional.mse_loss(pred, batch.target)
        absmax_error = (pred - batch.target).abs().max()
        
        self.log("val/data_loss", data_loss)
        self.log("val/cls_loss", cls_loss)
        self.log("val/unweighted_router_loss", router_loss_total)
        self.log("val/loss", loss)
        self.log("val/cls_pred_correct", cls_pred_correct)
        self.log("val/mae_loss", mae_loss)
        self.log("val/mse_loss", mse_loss)
        self.log("val/absmax_error", absmax_error)
        return loss
    
@hydra.main(version_base=None, config_path="../../../config", config_name="moe_dpot")
def main(cfg: DictConfig) -> None:
    
    print(cfg)

    train_dataset = InMemForecastDataset(
        filenames=cfg.data_cfg.train_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        history_time_window=cfg.model_cfg.in_timesteps,
        future_time_window=cfg.model_cfg.out_timesteps,
        time_step=cfg.data_cfg.time_step,
        start_time=cfg.data_cfg.start_time,
        normalizer=None,
        augment=True,
        layout=cfg.layout
    )
    val_dataset = InMemForecastDataset(
        filenames=cfg.data_cfg.val_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        history_time_window=cfg.model_cfg.in_timesteps,
        future_time_window=cfg.model_cfg.out_timesteps,
        time_step=cfg.data_cfg.time_step,
        start_time=cfg.data_cfg.start_time,
        normalizer=None,
        augment=False,
        layout=cfg.layout
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4 * cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate,
    )
    
    # Setup Wandb Logger.
    log_id_parts = [
        "moe_dpot",
        cfg.data_cfg.dataset.lower(),
        date.today().strftime("%Y-%m-%d"),
    ]
    if os.getenv("SLURM_JOB_ID") is not None:
        log_id_parts.append(os.getenv("SLURM_JOB_ID"))
    
    log_id = "_".join(log_id_parts)
    cfg.log_dir = os.path.join(cfg.log_dir, log_id)
    os.makedirs(cfg.log_dir, exist_ok=True)
    
    logger = WandbLogger(
        entity="hpcforge",
        project="bubbleformer",
        name=log_id,
        dir=cfg.log_dir,
        config=OmegaConf.to_container(cfg),
    )
    
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.devices,
        num_nodes=cfg.nodes,
        strategy="auto",
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=100,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        logger=logger,
        default_root_dir=cfg.log_dir,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        callbacks=[
            ModelSummary(max_depth=-1), 
            ModelCheckpoint(
                dirpath=cfg.log_dir + "/checkpoints",
                monitor="val/loss",
                mode="min",
                save_top_k=2,
                save_last=True,
                every_n_train_steps=20000,
                save_on_exception=True
            ),
        ]
    )
    
    model = MoEPOTNet(
        OmegaConf.to_container(cfg.model_cfg),
        cfg.router_loss_weight,
        cfg.optim_cfg.lr
    )
    
    total_params = count_model_parameters(model, active=False)
    print(f"Total Model parameters: {total_params:,d}")

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()