import torch
import pytest 

def parametrize_available_devices(param_name: str):
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return pytest.mark.parametrize(param_name, devices)