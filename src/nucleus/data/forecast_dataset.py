from typing import List, Optional
import json
import random
import numpy as np
import h5py as h5
import torch
from torch.utils.data import Dataset
from nucleus.data.batching import make_data
from nucleus.data.normalize import Normalizer
from nucleus.data.layout import convert_layout


class ForecastDatasetBase(Dataset):
    def __init__(
        self,
        filenames: List[str],
        input_fields: Optional[List[str]],
        output_fields: Optional[List[str]],
        time_step: int,
        start_time: int,
        normalizer: Optional[Normalizer],
        augment: bool,
        layout: str = "t h w c",
    ):
        super().__init__()
        self.filenames = filenames
        self.input_fields = input_fields if input_fields is not None else ["dfun", "temperature", "velx", "vely"]
        self.output_fields = output_fields if output_fields is not None else ["dfun", "temperature", "velx", "vely"]
        self.time_step = time_step
        self.start_time = start_time
        self.normalizer = normalizer
        self.augment = augment
        self.layout = layout
        self.data = None

        fluid_params_files = [fname.replace(".hdf5", ".json") for fname in filenames]
        self.fluid_params = []
        for fluid_params_file in fluid_params_files:
            with open(fluid_params_file, "r", encoding="utf-8") as f:
                self.fluid_params.append(json.load(f))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data'] = None
        state.pop('num_trajs', None)
        state.pop('traj_lens', None)
        return state

    def _ensure_open(self):
        if getattr(self, "data", None) is None:
            self.data = [h5.File(filename, "r") for filename in self.filenames]
            self.num_trajs = []
            self.traj_lens = []
            for h5_file in self.data:
                self.num_trajs.append(1)
                self.traj_lens.append(h5_file[self.input_fields[0]].shape[0])

    def _get_traj_len(self, traj_len: int) -> int:
        raise NotImplementedError

    def __len__(self):
        self._ensure_open()
        total_len = 0
        for (num_traj, traj_len) in zip(self.num_trajs, self.traj_lens):
            total_len += num_traj * self._get_traj_len(traj_len)
        return total_len

    def _index_to_file_start(self, idx: int):
        samples_per_traj = [x * self._get_traj_len(y) for x, y in zip(self.num_trajs, self.traj_lens)]
        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        start = idx + self.start_time - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)
        return file_idx, start


class ForecastDataset(ForecastDatasetBase):
    def __init__(
        self,
        filenames: List[str],
        input_fields: Optional[List[str]],
        output_fields: Optional[List[str]],
        future_time_window: int,
        history_time_window: int,
        time_step: int,
        start_time: int,
        fluid_params: List[str],
        heater_params: List[str],
        global_params: List[str],
        layout: str,
        normalizer: Optional[Normalizer],
        augment: bool,
    ):
        super().__init__(
            filenames=filenames,
            input_fields=input_fields,
            output_fields=output_fields,
            time_step=time_step,
            start_time=start_time,
            normalizer=normalizer,
            augment=augment,
            layout=layout,
        )
        self.future_time_window = future_time_window
        self.history_time_window = history_time_window
        self.time_step = time_step
        self.start_time = start_time
        self.fluid_params = fluid_params
        self.heater_params = heater_params
        self.global_params = global_params
        self.layout = layout  
        self.normalizer = normalizer
        self.augment = augment
        
        self.data = None

        self.input_num_fields = len(self.input_fields)
        self.output_num_fields = len(self.output_fields)
        self.fields = list(set(self.input_fields + self.output_fields))

        self.diff_terms = {k:[] for k in self.fields}
        self.div_terms = {k:[] for k in self.fields}

        sim_params_files = [fname.replace(".hdf5", ".json") for fname in filenames]
        self.sim_params = []
        for sim_params_file in sim_params_files:
            with open(sim_params_file, "r", encoding="utf-8") as f:
                sim_params_json = json.load(f)
            self.sim_params.append(sim_params_json)

    def _get_traj_len(self, traj_len: int) -> int:
        return traj_len - self.start_time - self.future_time_window - self.history_time_window + 1

    def __getitem__(self, idx: int):
        self._ensure_open()
        file_idx, start = self._index_to_file_start(idx)

        inp_slice = slice(start, start + self.history_time_window, self.time_step)
        out_slice = slice(
            start + self.history_time_window,
            start + self.history_time_window + self.future_time_window,
            self.time_step,
        )

        inp_data = torch.stack(
            [torch.from_numpy(np.array(self.data[file_idx][f][inp_slice])) for f in self.input_fields],
            dim=-1,
        )
        out_data = torch.stack(
            [torch.from_numpy(np.array(self.data[file_idx][f][out_slice])) for f in self.output_fields],
            dim=-1,
        )
        
        sim_params = self.sim_params[file_idx]
        bulk_temp = int(sim_params["bulk_temp"])

        if self.normalizer is not None:
            inp_data = self.normalizer.normalize(inp_data, bulk_temp)
            out_data = self.normalizer.normalize(out_data, bulk_temp)
            sim_params = self.normalizer.normalize_params([sim_params])[0]
        
        if self.augment:
            if random.random() < 0.5:
                # [T H W C], we flip along the width (dim=2)
                inp_data = torch.flip(inp_data, dims=[2])
                out_data = torch.flip(out_data, dims=[2])
                
        inp_data = convert_layout(inp_data, self.layout)
        out_data = convert_layout(out_data, self.layout)

        return make_data(
            input=inp_data.float(),
            target=out_data.float(),
            sim_params_dict=sim_params,
            downsample_factor=1,
            fluid_params=self.fluid_params,
            heater_params=self.heater_params,
            global_params=self.global_params
        )