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
from nucleus.data.normalize import get_normalizer
from nucleus.trajectory import Trajectory

FIELDS = ["dfun", "temperature", "velx", "vely"]

@pytest.mark.parametrize("model_name", ["nucleus2_moe", "nucleus2_moe_divfree"])
@pytest.mark.parametrize("trajectory_steps", [8, 16, 20, 24, 30, 31, 32])
def test_run(
    model_name,
    trajectory_steps
):
    # The divfree model needs the divfree-specific normalization stats (sdf/velx/
    # vely/temp); the default `standard` normalizer doesn't carry them.
    overrides = ["normalizer_cfg=divfree"] if model_name == "nucleus2_moe_divfree" else []
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="inference", overrides=overrides)

    cfg.start_time = 0
    
    cfg.model_cfg.name = model_name
    model_kwargs = OmegaConf.to_container(cfg.model_cfg.params, resolve=True)
    model_kwargs["input_fields"] = 4
    model_kwargs["output_fields"] = 4
    model_kwargs["embed_dim"] = 32
    model_kwargs["num_heads"] = 1
    model_kwargs["mlp_ratio"] = 1
    model_kwargs["processor_blocks"] = 1
    model_kwargs["num_experts"] = 1
    model_kwargs["topk"] = 1
    model_kwargs["patching"] = "Linear"
    model = get_model(cfg.model_cfg.name, **model_kwargs)
    model = model.to('cpu')
    
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sim.hdf5")
        with h5py.File(path, "w") as handle:
            for field in FIELDS:
                handle.create_dataset(field, data=np.random.randn(50, 64, 64).astype(np.float32))
            # Staggered face velocities the divfree model reads on their natural grids.
            handle.create_dataset("velfacex", data=np.random.randn(50, 64, 65).astype(np.float32))
            handle.create_dataset("velfacey", data=np.random.randn(50, 65, 64).astype(np.float32))
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
    
        normalizer = get_normalizer(cfg.normalizer_cfg)

        test_results: TestResults = run_test(
            cfg, 
            model, 
            normalizer, 
            test_file_path=path, 
            trajectory_steps=trajectory_steps
        )
        # Divfree results are natural-grid Trajectories; collocated results are
        # (T, H, W, 4) tensors. Compare field-by-field so both are handled.
        def fields(result):
            if isinstance(result, Trajectory):
                return [result.sdf, result.temp, result.velx, result.vely]
            return [result]

        for pred_field, target_field in zip(fields(test_results.preds), fields(test_results.targets)):
            assert pred_field.isfinite().all()
            assert target_field.isfinite().all()
            assert pred_field.shape[1:] == target_field.shape[1:]