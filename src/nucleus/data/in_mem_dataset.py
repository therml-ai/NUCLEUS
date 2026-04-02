from typing import List, Optional, Tuple, Dict
import json
import dataclasses
import numpy as np
import h5py as h5
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from nucleus.data.batching import Data
from nucleus.data.batching import make_data
from nucleus.data.normalize import Normalizer

class InMemDataset(Dataset):
    """
    Dataset class for time series forecasting on the BubbleML dataset.
    This downsamples the full dataset and stores it in cpu memory. This can be
    used to accelerate data loading on a networked file system.
    """
    def __init__(
        self,
        filenames: List[str],
        input_fields: Optional[List[str]],
        output_fields: Optional[List[str]],
        future_time_window: int,
        history_time_window: int,
        time_step: int,
        start_time: int,
        normalizer: Optional[Normalizer],
        augment: bool = False,
    ):
        super().__init__()
        self.filenames = filenames
        
        self.future_time_window = future_time_window
        self.history_time_window = history_time_window
        self.time_step = time_step
        self.start_time = start_time
        self.normalizer = normalizer
        self.augment = augment
        
        if input_fields is not None:
            self.input_fields = input_fields
        else:
            self.input_fields = ["dfun", "temperature", "velx", "vely"]
        if output_fields is not None:
            self.output_fields = output_fields
        else:
            self.output_fields = ["dfun", "temperature", "velx", "vely"]
        self.fields = list(set(self.input_fields + self.output_fields))
        
        self.data = self._load_data()
        self.num_trajs = [1 for _ in range(len(self.filenames))]
        self.traj_lens = [d[self.input_fields[0]].shape[0] for d in self.data]

        self.diff_terms = {k:[] for k in self.fields}
        self.div_terms = {k:[] for k in self.fields}


        fluid_params_files = [fname.replace(".hdf5", ".json") for fname in filenames]
        self.fluid_params = []
        for fluid_params_file in fluid_params_files:
            with open(fluid_params_file, "r", encoding="utf-8") as f:
                fluid_params = json.load(f)
            self.fluid_params.append(fluid_params)

    def _load_data(self):
        hdf5_files = [h5.File(filename, "r") for filename in self.filenames]
        data = []
        for hdf5_file in hdf5_files:
            d = {}
            for field in self.fields:
                field_data = torch.tensor(hdf5_file[field][...])
                d[field] = field_data
            data.append(d)
        return data
    
    def _get_traj_len(self, traj_len: int) -> int:
        return traj_len - self.start_time - self.future_time_window - self.history_time_window + 1

    def __len__(self):
        total_len = 0
        for (num_traj, traj_len) in zip(self.num_trajs, self.traj_lens):
            total_len += num_traj * self._get_traj_len(traj_len)
        return total_len

    def __getitem__(self, idx: int) -> Data:
        samples_per_traj = [
            x * self._get_traj_len(y)
            for x, y in zip(self.num_trajs, self.traj_lens)
        ]

        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        start = idx + self.start_time - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)

        inp_slice = slice(start, start + self.history_time_window, self.time_step)
        out_slice = slice(
            start + self.history_time_window, 
            start + self.history_time_window + self.future_time_window, 
            self.time_step
        )

        inp_data = []
        out_data = []

        for field in self.input_fields:
            data_item = torch.tensor(self.data[file_idx][field][inp_slice])                
            inp_data.append(data_item)
        for field in self.output_fields:
            data_item = torch.tensor(self.data[file_idx][field][out_slice])
            out_data.append(data_item)

        inp_data = torch.stack(inp_data, dim=-1) # (T, H, W, C)
        out_data = torch.stack(out_data, dim=-1) # (T, H, W, C)
        
        fluid_params = self.fluid_params[file_idx]
        bulk_temp = int(fluid_params["bulk_temp"])
        
        if self.normalizer is not None:
            inp_data = self.normalizer.normalize(inp_data, bulk_temp)
            out_data = self.normalizer.normalize(out_data, bulk_temp)
            fluid_params = self.normalizer.normalize_params([fluid_params])[0]
        
        if self.augment:
            if random.random() < 0.5:
                # [T H W C], we flip along the width (dim=2)
                inp_data = torch.flip(inp_data, dims=[2])
                out_data = torch.flip(out_data, dims=[2])

        return make_data(
            input=inp_data.float(),
            target=out_data.float(),
            fluid_params_dict=fluid_params,
            downsample_factor=1
        )