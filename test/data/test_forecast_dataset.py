import h5py
import json
import numpy as np
import os
import pytest
import tempfile

from nucleus.data.forecast_dataset import ForecastDataset
from nucleus.data.in_mem_forecast_dataset import InMemForecastDataset
from nucleus.data.layout import channel_dim, time_dim

FIELDS = ["dfun", "temperature", "velx", "vely"]

fluid_params = {
    "val1": 1,
    "val2": 2
}
heater_params = {
    "hot1": 1,
    "hot2": 2
}
global_params = {
    "g1": 1,
    "g2": 2
}

@pytest.mark.parametrize("dataset_class", [ForecastDataset, InMemForecastDataset])
@pytest.mark.parametrize("history_time_window", [1, 2, 8, 16])
@pytest.mark.parametrize("future_time_window", [1, 2, 8, 16])
@pytest.mark.parametrize("layout", ["t h w c", "t c h w", "h w t c"])
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
            params.update(fluid_params)
            params.update({"heater": heater_params})
            params.update(global_params)
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
            fluid_params,
            heater_params,
            global_params,
            layout,
            None,
            True
        )
        
        for i in range(3):
            data = dataset[i]
            assert data.input.shape[time_dim(layout)] == history_time_window
            assert data.target.shape[time_dim(layout)] == future_time_window
            assert data.input.shape[channel_dim(layout)] == 4
            assert len(data.sim_params_tensor) == len(fluid_params) + len(heater_params) + len(global_params)
            assert data.sim_params_tensor[0] == fluid_params["val1"]