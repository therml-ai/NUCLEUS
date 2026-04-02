import numpy as np

def heatflux(
        dfun: np.ndarray,
        temp: np.ndarray,
        heater_temp: int
    ):
    """
    Calculates heatflux for FC-72 fluid.
    To-Do:
        Generalize for other fluids
    Args:
        dfun: T, H, W np.ndarray
        temp: T, H, W np.ndarrray
        heater_temp: int
    """
    dx = 1/32
    lc = 0.0007

    x_min, x_max = -8, 8
    y_min, y_max =  0, 16
    nx = int((x_max - x_min) / dx)  # Should be 512
    ny = int((y_max - y_min) / dx)  # Should be 512

    x_centers = x_min + (np.arange(nx) + 0.5)*dx
    y_centers = y_min + (np.arange(ny) + 0.5)*dx

    x_grid, _ = np.meshgrid(x_centers, y_centers)

    heater_mask = (x_grid >= -5.0) & (x_grid <= 5.0)    # 512, 512
    heater_mask_3d = np.broadcast_to(heater_mask, (dfun.shape[0], 512, 512)) # T, 512, 512

    liquid_mask = dfun < 0    # T, 512, 512
    temp_fields = (heater_mask_3d & liquid_mask).astype(float) * (heater_temp - temp) # T, 512, 512
    hflux_fields = 0.054 * (temp_fields / (dx * lc))
    hfluxes = hflux_fields[:, 0, :].mean(axis=1)

    return np.mean(hfluxes), np.max(hfluxes)
