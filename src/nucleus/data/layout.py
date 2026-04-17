import torch
import einops

THWC_LAYOUT = "t h w c"
HWTC_LAYOUT = "h w t c"
TCHW_LAYOUT = "t c h w"

_LAYOUTS = [
    "t h w c", # Ideal for neighborhood attention, MLPs
    "h w t c",
    "t c h w" # used by bubbleformer and convs
]

def convert_layout(data: torch.Tensor, target_layout: str, source_layout: str = "t h w c") -> torch.Tensor:
    assert target_layout in _LAYOUTS, f"Invalid target layout: {target_layout}"
    assert source_layout in _LAYOUTS, f"Invalid source layout: {source_layout}"
    assert data.dim() >= 4, f"Data must have at least 4 dimensions, got {data.dim()}"
    return einops.rearrange(data, f"... {source_layout} -> ... {target_layout}")

def channel_dim(layout: str):
    assert "c" in layout
    return layout.split(" ").index("c")

def index_channel_dim(data: torch.Tensor, layout: str, index: int):
    assert layout in _LAYOUTS, f"invalid layout {layout} must be from {_LAYOUTS}"
    dim_to_index = channel_dim(layout) + 1
    return torch.index_select(data, dim=dim_to_index, index=index).squeeze(dim_to_index)

def get_sdf(data: torch.Tensor, layout: str):
    return index_channel_dim(data, layout, torch.tensor([0], dtype=torch.int64, device=data.device))

def get_temp(data: torch.Tensor, layout: str):
    return index_channel_dim(data, layout, torch.tensor([1], dtype=torch.int64, device=data.device))

def get_velx(data: torch.Tensor, layout: str):
    return index_channel_dim(data, layout, torch.tensor([2], dtype=torch.int64, device=data.device))

def get_vely(data: torch.Tensor, layout: str):
    return index_channel_dim(data, layout, torch.tensor([3], dtype=torch.int64, device=data.device))