import torch

def upsample(tensor, scale_factor):
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    assert tensor.dim() == 4, "interpolated tensor must be of shape (B, C, H, W)"

    return torch.nn.functional.interpolate(tensor, scale_factor=scale_factor, mode="bicubic").squeeze()

def downsample(tensor, scale_factor):
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    assert tensor.dim() == 4, "interpolated tensor must be of shape (B, C, H, W)"

    return torch.nn.functional.interpolate(tensor, scale_factor=1 / scale_factor, mode="bicubic").squeeze()
