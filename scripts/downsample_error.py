import argparse
import numpy as np
import h5py
from pathlib import Path
import torch

from boiling_viz.boiling_video import BoilingVideoBuilder


def load_trajectory(path):
    FIELDS = ["dfun", "temperature", "velx", "vely"]
    with h5py.File(path, "r") as handle:
        return np.stack([handle[field] for field in FIELDS], axis=0)

def vapor_volume(sdf, dx, dy):
    vapor_mask = sdf > 0
    return (vapor_mask.to(torch.float32).sum(axis=(-2, -1)) * dx * dy).mean()

def temp_above_heater(temp, x_grid, dx, dy):
    on_heater = ((x_grid >= -5.25) & (x_grid <= 5.25)).to(temp.device)
    num_on_heater = on_heater.sum()
    return ((temp[:, 0] * on_heater).sum(dim=-1) / num_on_heater)

def heat_flux(temp, sdf, heater_temp, dx, dy):
    def denormalize_temp_grad(temp, t_wall, t_bulk=50, k=0.054):
        del_t = t_wall - t_bulk
        return 2 * k * del_t * (1 - temp)
    
    def non_dim_temp(temp, bulk_temp=50):
        return (temp - bulk_temp) / (heater_temp - bulk_temp)

    lc = 0.0007
    x_grid = torch.arange(-8, 8, dx) + dx / 2

    liquid_mask = sdf < 0 #temp < 58

    temp = non_dim_temp(temp)
    d_temp = denormalize_temp_grad(temp[:, 0], heater_temp)
    heater_mask = (x_grid >= -5.25) & (x_grid <= 5.25)
    hflux_list = torch.mean((heater_mask[None, :] & liquid_mask[:, 0]).to(float) * d_temp / (dy * lc),
                            dim=1)
    hflux = torch.mean(hflux_list)
    qmax = torch.max(hflux_list)
    return hflux, qmax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str)
    args = parser.parse_args()
    path = Path(args.path)
    traj = load_trajectory(path)
    
    traj = torch.from_numpy(traj)
    
    print(traj.shape)
        
    down = traj[:, :, 0::4, ::8] # downsample starting from row above heater.
    
    # Technically for upsampling it's more important to have the boundary,
    # since here it's basically interpolating with ghost cells.
    traj_recon = torch.nn.functional.interpolate(down, (512, 512), mode="bicubic")

    print(heat_flux(traj[1], traj[0], 97, 1/32, 1/32))
    print(heat_flux(down[1], down[0], 97, 1/4, 1/32)) # Use dy=1/32 since gap to heater unchanged
    print(heat_flux(traj_recon[1], traj_recon[0], 97, 1/32, 1/32))
    

if __name__ == "__main__":
    main()