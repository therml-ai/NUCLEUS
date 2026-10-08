import pytest
import torch

from nucleus.data.normalize import NoNormalizer, get_normalizer


def test_no_normalizer_requires_no_constants():
    normalizer = get_normalizer({"name": "no"})
    assert isinstance(normalizer, NoNormalizer)
    parameters = [{"bulk_temp": 87.0}]
    assert normalizer.normalize_params(parameters) is parameters
    assert normalizer.unnormalize_params(parameters) is parameters


@pytest.mark.parametrize("layout", ["t h w c", "t c h w"])
def test_no_normalizer_preserves_fields_with_layout(layout):
    normalizer = get_normalizer({"name": "no"})
    field_values = torch.tensor([-2.5, 0.0, 87.0])
    bulk_temp = torch.tensor(87.0)
    assert normalizer.normalize(field_values, bulk_temp, layout=layout) is field_values
    assert normalizer.unnormalize(field_values, bulk_temp, layout=layout) is field_values
