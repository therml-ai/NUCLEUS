from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from nucleus.data.layout import convert_layout
from nucleus.data.normalize import get_normalizer


@pytest.mark.parametrize("channel_count", [4, 6])
@pytest.mark.parametrize("layout", ["t h w c", "t c h w"])
def test_divfree_normalization_fields_and_round_trip(channel_count, layout):
    config_path = Path(__file__).resolve().parents[2] / "config/normalizer_cfg/divfree.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    normalizer = get_normalizer(config)
    fields = torch.randn(2, 3, 4, 5, channel_count)
    bulk_temperature = torch.tensor([58.0, 62.0])
    expected = torch.stack([
        normalizer.normalize_sdf(fields[..., 0]),
        normalizer.normalize_temp(fields[..., 1], bulk_temperature),
        normalizer.normalize_velx(fields[..., 2]),
        normalizer.normalize_vely(fields[..., 3]),
    ], dim=-1)
    original = convert_layout(fields, target_layout=layout).clone()
    normalized = normalizer.normalize(original, bulk_temperature, layout=layout)
    canonical = convert_layout(normalized, target_layout="t h w c", source_layout=layout)
    torch.testing.assert_close(canonical[..., :4], expected)
    if channel_count == 6:
        torch.testing.assert_close(canonical[..., 4], normalizer.normalize_psi(fields[..., 4]))
        torch.testing.assert_close(canonical[..., 5], normalizer.normalize_phi(fields[..., 5]))
    restored = normalizer.unnormalize(normalized, bulk_temperature, layout=layout)
    torch.testing.assert_close(restored, original, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(original, convert_layout(fields, target_layout=layout))
