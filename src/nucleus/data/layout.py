import torch
import einops

_LAYOUTS = [
    "t h w c", #  Ideal for neighborhood attention, MLPs
    "h w t c",
    "t c h w" # used by bubbleformer and convs
]

def convert_layout(data: torch.Tensor, layout: str) -> torch.Tensor:
    # By default, data is assumed to be in a (t, h, w, c) layout.
    assert layout in _LAYOUTS, f"Invalid layout: {layout}"
    assert data.dim() == 4, f"Data must have 4 dimensions, got {data.dim()}"
    return einops.rearrange(data, f"t h w c -> {layout}")