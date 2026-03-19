from typing import List, Optional, Tuple, Dict
import json

import numpy as np
import h5py as h5
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from bubbleformer.data.batching import make_data

class BubbleForecast(Dataset):
    """
    Dataset class for time series forecasting on the BubbleML dataset
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
        norm: str = "none",
        downsample_factor: int = 1,
        return_fluid_params: bool = False,
    ):
        super().__init__()
        self.filenames = filenames
        if input_fields is not None:
            self.input_fields = input_fields
        else:
            self.input_fields = ["dfun", "temperature", "velx", "vely"]
        if output_fields is not None:
            self.output_fields = output_fields
        else:
            self.output_fields = ["dfun", "temperature", "velx", "vely"]
        self.norm = norm
        self.downsample_factor = downsample_factor
        #self.time_window = time_window
        self.future_time_window = future_time_window
        self.history_time_window = history_time_window
        self.time_step = time_step
        self.start_time = start_time
        
        self.data = [h5.File(filename, "r") for filename in filenames]
        self.num_trajs = []
        self.traj_lens = []

        for h5_file in self.data:
            self.num_trajs.append(1)
            self.traj_lens.append(h5_file[self.input_fields[0]].shape[0])

        self.input_num_fields = len(self.input_fields)
        self.output_num_fields = len(self.output_fields)
        self.fields = list(set(self.input_fields + self.output_fields))
        self.diff_terms = {k:[] for k in self.fields}
        self.div_terms = {k:[] for k in self.fields}

        self.return_fluid_params = return_fluid_params
        if self.return_fluid_params:
            fluid_params_files = [fname.replace(".hdf5", ".json") for fname in filenames]
            self.fluid_params = []
            for fluid_params_file in fluid_params_files:
                with open(fluid_params_file, "r", encoding="utf-8") as f:
                    fluid_params = json.load(f)
                self.fluid_params.append(fluid_params)

    def _get_traj_len(self, traj_len: int) -> int:
        return traj_len - self.start_time - self.future_time_window - self.history_time_window + 1

    def __len__(self):
        total_len = 0
        for (num_traj, traj_len) in zip(self.num_trajs, self.traj_lens):
            total_len += num_traj * self._get_traj_len(traj_len)
        return total_len

    def __getitem__(self, idx: int):
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
            if self.downsample_factor > 1:
                _, h, w = data_item.shape
                new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                data_item = F.interpolate(
                    data_item.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="bilinear"
                ).squeeze(1)
                
            inp_data.append(data_item)
        for field in self.output_fields:
            data_item = torch.tensor(self.data[file_idx][field][out_slice])
            if self.downsample_factor > 1:
                _, h, w = data_item.shape
                new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
                data_item = F.interpolate(
                    data_item.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="bilinear"
                ).squeeze(1)
            out_data.append(data_item)

        inp_data = torch.stack(inp_data, dim=-1)
        out_data = torch.stack(out_data, dim=-1)

        return make_data(
            input=inp_data.float(),
            target=out_data.float(),
            fluid_params_dict=self.fluid_params[file_idx],
            downsample_factor=self.downsample_factor
        )