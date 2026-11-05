import torch
from scipy.ndimage import gaussian_filter

from nucleus.utils.sdf_smoothing import smooth_sdf_frames


def test_blur_only_matches_gaussian_without_reinitialization(monkeypatch):
    def unexpected_reinitialization(*args, **kwargs):
        raise AssertionError("Blur-only output must precede reinitialization")

    monkeypatch.setattr(
        "nucleus.utils.sdf_smoothing.sdf_reinit_sussman",
        unexpected_reinitialization,
    )
    sdf = torch.arange(32).float().reshape(2, 4, 4)
    actual = smooth_sdf_frames(sdf, 2.0, dx=1.0, dy=1.0, reinitialize=False)
    expected = gaussian_filter(sdf.numpy(), sigma=(0, 2.0, 2.0), mode="reflect")
    assert torch.allclose(actual, torch.from_numpy(expected))


def test_smoothing_reduces_noise_without_mixing_frames_or_mutating_input():
    checkerboard = (torch.arange(8)[:, None] + torch.arange(8)[None, :]) % 2
    noisy_frame = checkerboard.float() * 0.2 - 0.1
    sdf = torch.stack([noisy_frame, torch.full_like(noisy_frame, 0.25)])
    original = sdf.clone()
    light = smooth_sdf_frames(sdf, 0.5, dx=1.0, dy=1.0)
    strong = smooth_sdf_frames(sdf, 1.0, dx=1.0, dy=1.0)
    assert strong[0].std() < light[0].std() < sdf[0].std()
    assert torch.allclose(light[1], sdf[1])
    assert torch.allclose(strong[1], sdf[1])
    assert torch.equal(sdf, original)
