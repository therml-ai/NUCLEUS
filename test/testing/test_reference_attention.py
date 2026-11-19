import natten
import pytest
import torch

from nucleus.testing.reference_attention import reference_natten, reference_neighborhood_attention


@pytest.mark.parametrize("shape, kernel_size", [
    ((2, 8, 6, 7, 2, 16), (8, 3, 3)),
    ((1, 5, 9, 9, 1, 32), (3, 5, 3)),
    ((2, 10, 11, 2, 16), (3, 5)),
])
def test_reference_attention_matches_natten(shape, kernel_size):
    generator = torch.Generator().manual_seed(0)
    query, key, value = (torch.randn(shape, generator=generator) for _ in range(3))
    natten_op = natten.na3d if len(kernel_size) == 3 else natten.na2d
    with torch.no_grad():
        expected = natten_op(query, key, value, kernel_size=kernel_size)
    actual = reference_neighborhood_attention(query, key, value, kernel_size=kernel_size)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_reference_natten_patches_and_restores():
    original = natten.na3d
    with reference_natten():
        assert natten.na3d is reference_neighborhood_attention
    assert natten.na3d is original
