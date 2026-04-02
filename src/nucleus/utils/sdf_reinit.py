r"""
This implements the fast marching method for reinitializing the SDF.
https://en.wikipedia.org/wiki/Fast_marching_method
This essentially assumes that the SDf is decently accurate near the interface
and less accurate away from the interface (which seems to be true in practice!)
"""
import torch
from torch.utils.cpp_extension import load

_sdf_reinit_cpp = load(
    "sdf_reinit",
    sources=["csrc/sdf_reinit.cpp"],
    extra_cflags=["-O3"],
    extra_link_args=["-fopenmp"],
    verbose=True
)

def sdf_reinit(sdf_init, dx, scale_factor=8, far_threshold=4.0):
    assert sdf_init.device == torch.device("cpu"), "SDF must be on CPU"
    return _sdf_reinit_cpp.sdf_reinit(sdf_init, dx, scale_factor, far_threshold)