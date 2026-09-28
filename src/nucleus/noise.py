import torch
import torch.nn.functional as F
import math
import random

class LogUniformNoise:
    def __init__(self, min, max, skip_prob):
        self.min = min
        self.max = max
        self.skip_prob = skip_prob # probability to NOT add noise.

    def __call__(self, tensor: torch.Tensor):
        assert tensor.dim() == 5
        log_scale = torch.rand(tensor.shape[0], 1, 1, 1, 1, device=tensor.device) * (math.log(self.max) - math.log(self.min)) + math.log(self.min)
        scale = log_scale.exp()
        noise = torch.randn_like(tensor) * scale
        skip = (torch.rand(tensor.shape[0], 1, 1, 1, 1, device=tensor.device) < self.skip_prob)
        return tensor + noise * (~skip).to(noise.dtype)


class FrameDropout:
    def __init__(self, p: float = 0.1, active_prob: float = 0.4):
        self.p = p
        self.active_prob = active_prob
        
    def __call__(self, x):
        b, t, _, _, c = x.shape
        active = (torch.rand(b, 1, 1, 1, c, device=x.device) < self.active_prob)
        mask = (torch.rand(b, t, 1, 1, c, device=x.device) > self.p).to(x.dtype)
        return x * torch.where(active, mask, torch.ones_like(mask))
    

class FieldDropout:
    def __init__(self, p: float = 0.1, active_prob: float = 0.4):
        self.p = p
        self.active_prob = active_prob
        
    def __call__(self, x):
        b, _, _, _, c = x.shape
        active = (torch.rand(b, 1, 1, 1, c, device=x.device) < self.active_prob)
        mask = (torch.rand(b, 1, 1, 1, c, device=x.device) > self.p).to(x.dtype)
        return x * torch.where(active, mask, torch.ones_like(mask))

def _disk_structuring_element(radius: int, device: torch.device) -> torch.Tensor:
    r"""
    Boolean disk of the given radius: cells whose Euclidean distance from the
    center is within `radius`. Used as an isotropic structuring element so
    dilation/erosion grows rounded regions rather than squares.
    """
    offsets = torch.arange(-radius, radius + 1, device=device)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    return (grid_y ** 2 + grid_x ** 2) <= radius ** 2


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    r"""
    Morphological dilation of a boolean mask (..., H, W) by a disk of the given
    radius. Out-of-bounds neighbors are treated as background, so dilation never
    grows past the domain.
    """
    spatial_shape = mask.shape[-2:]
    disk = _disk_structuring_element(radius, mask.device).to(torch.float32)
    kernel = disk.reshape(1, 1, *disk.shape)
    flattened = mask.to(torch.float32).reshape(-1, 1, *spatial_shape)
    overlap = F.conv2d(flattened, kernel, padding=radius)
    return (overlap > 0).reshape(mask.shape)


class PhaseAugmentation:
    r"""
    Base class for perturbations of a binary phase mask (0 - liquid, nonzero -
    vapor). These make the model robust to the phase errors it produces during
    autoregressive rollout. Subclasses implement `_perturb` on a boolean vapor
    mask; this base handles the per-call activation draw and the dtype round-trip.
    """
    def __init__(self, active_prob: float = 0.4):
        self.active_prob = active_prob

    def __call__(self, phase: torch.Tensor) -> torch.Tensor:
        assert phase.dim() == 4, "phase must be (B, T, H, W)"
        if random.random() >= self.active_prob:
            return phase
        vapor = self._perturb(phase != 0)
        return vapor.to(phase.dtype)

    def _perturb(self, vapor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SpuriousBulkNucleation(PhaseAugmentation):
    r"""
    Rare per-cell liquid -> vapor flips far from any bubble, mimicking spurious
    vapor specks the model should learn to erase.
    """
    def __init__(
        self,
        active_prob: float = 0.1,
        bulk_flip_prob: float = 1e-4, # 64 * 64 * 8 * 1e-4 ~= 3 per batch item
        bulk_margin: int = 2,
    ):
        super().__init__(active_prob)
        self.bulk_flip_prob = bulk_flip_prob
        self.bulk_margin = bulk_margin

    def _perturb(self, vapor: torch.Tensor) -> torch.Tensor:
        bulk_liquid = (~vapor) & ~_dilate_mask(vapor, self.bulk_margin)
        flips = bulk_liquid & (torch.rand_like(vapor, dtype=torch.float32) < self.bulk_flip_prob)
        return vapor | flips


class InterfaceJitter(PhaseAugmentation):
    r"""
    Per-cell flips in a band straddling the interface, which roughens its
    position. Bubble interiors are left untouched.
    """
    def __init__(
        self,
        active_prob: float = 0.2,
        interface_flip_prob: float = 0.05,
        interface_radius: int = 1,
    ):
        super().__init__(active_prob)
        self.interface_flip_prob = interface_flip_prob
        self.interface_radius = interface_radius

    def _perturb(self, vapor: torch.Tensor) -> torch.Tensor:
        # cells within interface_radius of BOTH a vapor and a liquid cell, i.e. the
        # band straddling the interface. Deep interiors are excluded.
        band = _dilate_mask(vapor, self.interface_radius) & _dilate_mask(~vapor, self.interface_radius)
        flips = band & (torch.rand_like(vapor, dtype=torch.float32) < self.interface_flip_prob)
        return vapor ^ flips


class BubbleResize(PhaseAugmentation):
    r"""
    Grows or shrinks every bubble by a small random radius, mimicking volume
    errors. Growing and shrinking are equally likely on each application.
    """
    def __init__(
        self,
        active_prob: float = 0.2,
        max_resize_radius: int = 1,
    ):
        super().__init__(active_prob)
        self.max_resize_radius = max_resize_radius

    def _perturb(self, vapor: torch.Tensor) -> torch.Tensor:
        radius = random.randint(1, self.max_resize_radius)
        if random.random() < 0.5:
            return _dilate_mask(vapor, radius)  # grow
        # eroding the vapor region is removing cells within radius of any liquid.
        return vapor & ~_dilate_mask(~vapor, radius)
