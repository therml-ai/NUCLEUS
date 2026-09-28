import torch


def vertical_interface(height, width, dx, dy, sat_temp, liquid_slope, vapor_slope):
    """A vertical interface with liquid (sdf < 0) on the left and vapor (sdf >= 0)
    on the right, each phase linear in x with its own slope and meeting sat_temp at
    the interface. The interface sits off any cell center, so no cell has sdf == 0,
    and the normal is +x, so grad(T).n equals the phase's slope.

    Returns ``(temp, sdf, x0)`` with the fields of shape ``(height, width)``."""
    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(height, dtype=torch.float64) + 0.5) * dy
    _, grid_x = torch.meshgrid(y, x, indexing="ij")
    x0 = x[width // 2] + 0.3 * dx
    sdf = grid_x - x0
    temp = torch.where(
        sdf >= 0,
        sat_temp + vapor_slope * (grid_x - x0),
        sat_temp + liquid_slope * (grid_x - x0),
    )
    return temp, sdf, float(x0)


def circular_interface(height, width, dx, dy, center, radius):
    """A bubble (sdf >= 0 inside) of the given radius, shape ``(height, width)``."""
    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(height, dtype=torch.float64) + 0.5) * dy
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return radius - torch.sqrt((grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2)

