import random
import numpy as np
import torch
from nucleus.data.batching import PushforwardData, make_pushforward_data
from nucleus.data.forecast_dataset import ForecastDataset
from nucleus.data.layout import convert_layout


class PushforwardForecastDataset(ForecastDataset):
    """
    Returns N contiguous windows for multi-step scheduled-sampling training.

    With num_windows=N, the time layout for a sample starting at `start` is:
        windows[0] : [start,               start + H)           — H frames (history)
        windows[k] : [start + H + (k-1)*F, start + H + k*F)     — F frames, for k = 1..N-1
        windows[-1] is the supervised target y.

    Requires history_time_window == future_time_window (H == F) so each model
    output can be fed directly as input to the next pass.

    num_windows=3 reproduces the original 1-step pushforward behaviour.
    """

    def __init__(self, *args, num_windows: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.history_time_window == self.future_time_window, (
            "PushforwardForecastDataset requires history_time_window == future_time_window"
        )
        assert num_windows >= 3, "num_windows must be >= 3 (at least x_prev, x_curr, y)"
        self.num_windows = num_windows

    def _get_traj_len(self, traj_len: int) -> int:
        # Needs H + (N-1)*F contiguous frames per sample.
        return traj_len - self.start_time - self.history_time_window - (self.num_windows - 1) * self.future_time_window + 1

    def __getitem__(self, idx: int) -> PushforwardData:
        self._ensure_open()

        samples_per_traj = [
            x * self._get_traj_len(y)
            for x, y in zip(self.num_trajs, self.traj_lens)
        ]
        cumulative_samples = np.cumsum(samples_per_traj)
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        start = idx + self.start_time - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)

        H = self.history_time_window
        F = self.future_time_window

        def _load(s):
            fields = []
            for field in self.input_fields:
                fields.append(torch.from_numpy(np.array(self.data[file_idx][field][s])))
            return torch.stack(fields, dim=-1)   # (T, H, W, C)

        # windows[0] = x_prev (history), windows[1..N-1] = successive F-frame windows
        slices = [slice(start, start + H, self.time_step)] + [
            slice(start + H + k * F, start + H + (k + 1) * F, self.time_step)
            for k in range(self.num_windows - 1)
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
