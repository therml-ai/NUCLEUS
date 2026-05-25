import h5py
import json
import numpy as np
import os
import pytest
import tempfile

from nucleus.data.forecast_dataset import ForecastDataset
from nucleus.data.in_mem_forecast_dataset import InMemForecastDataset
from nucleus.data.layout import channel_dim

FIELDS = ["dfun", "temperature", "velx", "vely"]

@pytest.mark.parametrize("dataset_class", [ForecastDataset, InMemForecastDataset])
@pytest.mark.parametrize("history_time_window", [1, 2, 8, 16])
@pytest.mark.parametrize("future_time_window", [1, 2, 8, 16])
@pytest.mark.parametrize("layout", ["t h w c", "t c h w"])
def test_in_mem_forecast_dataset(
    dataset_class,
    history_time_window,
    future_time_window,
    layout
):
    test_path = "sim.hdf5"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, test_path)
        with h5py.File(path, "w") as handle:
            for field in FIELDS:
                handle.create_dataset(field, data=np.random.randn(100, 64, 64))
        json_path = path.replace("hdf5", "json")
        with open(json_path, "w") as handle:
            params = dict(
                bulk_temp=50,
                sat_temp=58,
                x_max=8,
                x_min=-8,
                y_max=16,
                y_min=0,
                num_blocks_x=24,
                num_blocks_y=24,
                nx_block=16,
                ny_block=16
            )
            json_params = json.dumps(params)
            handle.write(json_params)
            
        dataset = dataset_class(
            [path],
            FIELDS,
            FIELDS,
            future_time_window,
            history_time_window,
            1,
            20,
            [],
            [],
            [],
            layout,
            None,
            True
        )
        
        assert dataset[0].input.shape[0] == history_time_window
        assert dataset[0].target.shape[0] == future_time_window
        assert dataset[0].input.shape[channel_dim(layout)] == 4