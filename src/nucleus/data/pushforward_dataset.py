import random
import numpy as np
import torch
from nucleus.data.batching import PushforwardData, make_pushforward_data
from nucleus.data.forecast_dataset import ForecastDatasetBase
from nucleus.data.layout import convert_layout


class PushforwardForecastDataset(ForecastDatasetBase):
    """
    Returns num_time_windows contiguous windows for scheduled-sampling training.

    Each window is time_window_size frames. windows[-1] is the supervised target;
    windows[0..N-2] are the ground-truth input windows.
    """

    def __init__(self, *args, num_time_windows: int = 3, time_window_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_time_windows >= 3, "num_time_windows must be >= 3 (at least x_prev, x_curr, y)"
        self.num_time_windows = num_time_windows
        self.time_window_size = time_window_size

    def _get_traj_len(self, traj_len: int) -> int:
        return traj_len - self.start_time - self.num_time_windows * self.time_window_size + 1

    def __getitem__(self, idx: int) -> PushforwardData:
        self._ensure_open()
        file_idx, start = self._index_to_file_start(idx)

        def _load(s):
            return torch.stack(
                [torch.from_numpy(np.array(self.data[file_idx][f][s])) for f in self.input_fields],
                dim=-1,
            )

        slices = [
            slice(start + k * self.time_window_size, start + (k + 1) * self.time_window_size, self.time_step)
            for k in range(self.num_time_windows)
        ]
        windows = [_load(s) for s in slices]

        fluid_params = self.fluid_params[file_idx]
        bulk_temp = int(fluid_params["bulk_temp"])

        if self.normalizer is not None:
            windows = [self.normalizer.normalize(w, bulk_temp) for w in windows]
            fluid_params = self.normalizer.normalize_params([fluid_params])[0]

        if self.augment and random.random() < 0.5:
            windows = [torch.flip(w, dims=[2]) for w in windows]

        windows = [convert_layout(w, self.layout).float() for w in windows]

        return make_pushforward_data(
            windows=windows,
            fluid_params_dict=fluid_params,
            downsample_factor=1,
        )
