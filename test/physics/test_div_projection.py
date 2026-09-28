import torch

from nucleus.physics.div_projection import div_projection
from nucleus.physics.poisson import divergence_centers_from_faces


def random_faces(height: int, width: int):
    generator = torch.Generator().manual_seed(0)
    facex = torch.randn(height, width + 1, generator=generator, dtype=torch.float64)
    facey = torch.randn(height + 1, width, generator=generator, dtype=torch.float64)
    # closed left/right/bottom walls carry no normal flux
    facex[:, 0] = 0.0
    facex[:, -1] = 0.0
    facey[0, :] = 0.0
    return facex, facey


def test_div_projection_without_source_is_divergence_free():
    dx, dy = 0.1, 0.1
    facex, facey = random_faces(32, 24)
    projx, projy = div_projection(facex, facey, dx, dy)
    divergence = divergence_centers_from_faces(projx, projy, dx, dy)
    assert torch.allclose(divergence, torch.zeros_like(divergence), atol=1e-8)


def test_div_projection_divergence_matches_source():
    dx, dy = 0.1, 0.1
    facex, facey = random_faces(32, 24)
    source = torch.randn(32, 24, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    projx, projy = div_projection(facex, facey, dx, dy, source=source)
    divergence = divergence_centers_from_faces(projx, projy, dx, dy)
    assert torch.allclose(divergence, source, atol=1e-8)
