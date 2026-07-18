import argparse
import numpy as np
import h5py
from pathlib import Path
import torch

from boiling_viz.boiling_video import BoilingVideoBuilder

def load_trajectory(path):
    FIELDS = ["dfun", "temperature", "velx", "vely"]
    with h5py.File(path, "r") as handle:
        return np.concatenate([handle[field] for field in FIELDS], axis=-1)


def vapor_volume(sdf, dx, dy):
    vapor_mask = sdf > 0
    return vapor_mask.astype(float).sum(axis=(-2, -1)) * dx * dy

def heat_flux(temp, sdf, heater_temp, dx, dy):
    def denormalize_temp_grad(temp, t_wall, t_bulk=50, k=0.054):
        del_t = t_wall - t_bulk
        return 2 * k * del_t * (1 - temp)
    
    def non_dim_temp(temp, bulk_temp=50):
        return (temp - bulk_temp) / (heater_temp - bulk_temp)

    lc = 0.0007
    x_grid = torch.arange(-8, 8, dx) + dx / 2

    print(temp.min(), temp.max())

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
    pred = load_trajectory(path / "pred_trajectory.hdf5")
    gt = load_trajectory(path / "gt_trajectory.hdf5")
    
    print(pred.shape)
    print(gt.shape)
        
    builder = BoilingVideoBuilder(pred[::10])
    builder.make_video(
        "./traj.gif", 
        duration=10, 
        colorbars=False,
        step_counter=True,
        field_titles=True,
        transparent_nan=False,
        columnwise=True
    )

if __name__ == "__main__":
    main()