import os
from hydra import initialize, compose
import numpy as np
from omegaconf import OmegaConf
import tempfile
import h5py
import random
import json
import pytest

from nucleus.run_forward_trajectory import run_test, TestResults
from nucleus.models import get_model
from nucleus.data.normalize import NoNormalizer

FIELDS = ["dfun", "temperature", "velx", "vely"]

@pytest.mark.parametrize("trajectory_steps", [8, 16, 20, 24, 30, 31, 32])
def test_run(
    trajectory_steps
):
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="default")

    model_kwargs = OmegaConf.to_container(cfg.model_cfg.params, resolve=True)
    model_kwargs["input_fields"] = 4
    model_kwargs["output_fields"] = 4
    model_kwargs["embed_dim"] = 32
    model_kwargs["num_heads"] = 1
    model_kwargs["mlp_ratio"] = 1
    model_kwargs["processor_blocks"] = 1
    model_kwargs["num_experts"] = 1
    model_kwargs["topk"] = 1
    model = get_model(cfg.model_cfg.name, **model_kwargs)
    model = model.to('cpu')
    
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sim.hdf5")
        with h5py.File(path, "w") as handle:
            for field in FIELDS:
                handle.create_dataset(field, data=np.random.randn(100, 64, 64).astype(np.float32))
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
            # these need to be set to pass into model, but value doesn't matter
            params.update(dict([(k, random.random()) for k in model.expected_fluid_params]))
            params.update({"heater": dict([(k, random.random()) for k in model.expected_heater_params])})
            params.update(dict([(k, random.random()) for k in model.expected_global_params]))
            params["setup"] = "subcooled"
            params["liquid"] = "fc72"
            json_params = json.dumps(params)
            handle.write(json_params)
    
        normalizer = NoNormalizer()

        test_results: TestResults = run_test(
            cfg, 
            model, 
            normalizer, 
            test_file_path=path, 
            trajectory_steps=trajectory_steps
        )
        assert test_results.preds.isfinite().all()
        assert test_results.targets.isfinite().all()
        assert test_results.preds.shape[1:] == test_results.targets.shape[1:]