from sympy.polys.domains.field import Field
import torch
import pytest
from nucleus.data.layout import (
    get_sdf,
    get_temp,
    get_velx,
    get_vely
)

@pytest.mark.parametrize("func", [get_sdf, get_temp, get_velx, get_vely])
def test_get_sdf_tchw(func):
    data = torch.randn(1, 5, 4, 64, 64)
    layout = "t c h w"
    field = func(data, layout)
    assert field.shape == torch.Size([1, 5, 64, 64])
    
@pytest.mark.parametrize("func", [get_sdf, get_temp, get_velx, get_vely])
def test_get_sdf_thwc(func):
    data = torch.randn(1, 5, 64, 64, 4)
    layout = "t h w c"
    field = func(data, layout)
    assert field.shape == torch.Size([1, 5, 64, 64])