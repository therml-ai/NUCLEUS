import torch
import hydra
from omegaconf import DictConfig
import numpy as np
import math
import h5py
import json
import matplotlib.pyplot as plt
import os

class DataDistribution:
    def __init__(self, bins, range):
        self.bins = np.linspace(range[0], range[1], bins)
        self.range = range
        self.hist = np.zeros(bins - 1)
        self.count = 0
        
    def update(self, value):
        self.hist += np.histogram(value, bins=self.bins, range=self.range)[0]
        self.count += value.size
        
    def var(self):
        bin_mid_points = (self.bins[1:] + self.bins[:-1]) / 2
        mean = np.average(bin_mid_points, weights=self.hist)
        return np.average((bin_mid_points - mean) ** 2, weights=self.hist).item()
    
    def std(self):
        return math.sqrt(self.var())

@hydra.main(config_path="../config", config_name="default")    
def main(cfg: DictConfig):
        
    # Initial loop to get the extrema for histogram bins
    max_sdf = float("-inf")
    max_temp = float("-inf")
    max_velx = float("-inf")
    max_vely = float("-inf")
    for train_path in cfg.data_cfg.train_paths:
        path = train_path.replace(".hdf5", ".json")
        with open(path, "r") as f:
            fluid_params_dict = json.load(f)
            heater_temp = fluid_params_dict["heater"]["wallTemp"]
            bulk_temp = fluid_params_dict["bulk_temp"]
            max_temp = max(max_temp, math.log1p(heater_temp - bulk_temp))
            #sdf = f["dfun"][:]
            #temp = f["temperature"][300::20]
            #velx = f["velx"][:]
            #vely = f["vely"][:]
            #max_sdf = max(max_sdf, np.abs(sdf).max().item())
            #max_temp = max(max_temp, np.abs(temp).max().item())
            #max_velx = max(max_velx, np.abs(velx).max().item())
            #max_vely = max(max_vely, np.abs(vely).max().item())
    
    #sdf_dist = DataDistribution(bins=4000, range=(-max_sdf, max_sdf))
    temp_dist = DataDistribution(bins=100, range=(-5, max_temp + 1))
    #velx_dist = DataDistribution(bins=4000, range=(-max_velx, max_velx))
    #vely_dist = DataDistribution(bins=4000, range=(-max_vely, max_vely))
    
    # Loop to get the normalization constants for the SDF, temperature, and velocities.
    for train_path in cfg.data_cfg.train_paths:
        #print(train_path)
        with h5py.File(train_path, "r") as f:
            #sdf = f["dfun"][500::10]
            temp = f["temperature"][300::10]
            #velx = f["velx"][500:]
            #vely = f["vely"][500:]
            
        with open(train_path.replace(".hdf5", ".json"), "r") as f:
            fluid_params_dict = json.load(f)
        
        heater_temp = fluid_params_dict["heater"]["wallTemp"]
        bulk_temp = fluid_params_dict["bulk_temp"]

        x_size = fluid_params_dict["x_max"] - fluid_params_dict["x_min"]
        y_size = fluid_params_dict["y_max"] - fluid_params_dict["y_min"]
        max_domain_size = max(x_size, y_size)
        
        min_temp = temp.min().item()
        max_temp = temp.max().item()
        
        if (min_temp < bulk_temp):
            print(train_path, "min_temp < bulk_temp:", min_temp, bulk_temp)
        if (max_temp > heater_temp):
            print(train_path, "max_temp > heater_temp:", max_temp, heater_temp)
        
        #print(heater_temp, bulk_temp, temp.min().item(), temp.max().item())
        
        #temp = np.clip(temp, a_min=bulk_temp, a_max=fluid_params_dict["heater"]["wallTemp"])
        #temp_dist.update(np.log1p(temp - bulk_temp))
        temp_dist.update(np.log1p(temp - bulk_temp))
        del temp
    
    print(temp_dist.hist.max())
    print(temp_dist.hist.mean())
    print(temp_dist.std())
    
    plt.bar(
        temp_dist.bins[:-1], 
        temp_dist.hist / temp_dist.hist.sum(), 
        width=temp_dist.bins[1] - temp_dist.bins[0]
    )
    plt.xlabel("Temperature")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.title("Temperature Distribution")
    plt.savefig("temp_hist_dist.png", bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()