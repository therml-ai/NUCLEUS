import torch
from scipy.ndimage import gaussian_filter

from nucleus.utils.sdf_reinit import sdf_reinit_sussman


def smooth_sdf_frames(
    sdf: torch.Tensor, sigma_cells: float, dx: float, dy: float,
    reinitialize: bool = True,
) -> torch.Tensor:
    """Smooth spatial dimensions independently, optionally reinitializing frames."""
    if sigma_cells <= 0:
        raise ValueError("sigma_cells must be positive.")
    filtered = gaussian_filter(
        sdf.detach().cpu().numpy(), sigma=(0, sigma_cells, sigma_cells), mode="reflect"
    )
    smoothed = torch.from_numpy(filtered).to(device=sdf.device, dtype=sdf.dtype)
    if not reinitialize:
        return smoothed
    return torch.stack([
        sdf_reinit_sussman(frame, dx=dx, dy=dy) for frame in smoothed
    ])
