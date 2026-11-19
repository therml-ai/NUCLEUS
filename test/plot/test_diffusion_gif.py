import pytest
import torch
from PIL import Image

from nucleus.data.layout import convert_layout
from nucleus.plot.diffusion_gif import save_diffusion_gif, save_rollout_gif


def test_rollout_gif_samples_frames_and_pads_sizes(tmp_path):
    frame_paths = []
    for frame_index, color in enumerate(["red", "green", "blue", "yellow"]):
        frame_path = tmp_path / f"rollout_{frame_index:04d}.png"
        Image.new("RGB", (10 + frame_index, 10), color).save(frame_path)
        frame_paths.append(frame_path)
    result = save_rollout_gif(
        frame_paths, tmp_path / "rollout.gif", max_frames=2, fps=5,
    )
    with Image.open(result) as animation:
        assert animation.n_frames == 2
        assert animation.size == (13, 10)
        assert animation.info["duration"] == 200
        assert animation.info["loop"] == 0
        assert animation.convert("RGB").getpixel((0, 0)) == (255, 0, 0)
        animation.seek(1)
        assert animation.convert("RGB").getpixel((0, 0)) == (255, 255, 0)


@pytest.mark.parametrize('layout', ['t h w c', 't c h w'])
@pytest.mark.parametrize('variant', [
    None, 'diffusion_gaussian_0_5', 'diffusion_gaussian_1_0',
    'diffusion_gaussian_2_0', 'diffusion_gaussian_3_0',
])
def test_comparison_gif_samples_multiple_windows(tmp_path, layout, variant):
    window_paths = []
    for window_index in range(2):
        sdf = torch.arange(24).reshape(2, 3, 4).float() - 12 + window_index
        fields = sdf[..., None].expand(-1, -1, -1, 4)
        window_path = tmp_path / f'window_{window_index}.pt'
        torch.save({
            'preds': convert_layout(fields, layout),
            'targets': convert_layout(fields, layout),
            'diffusion_raw_sdf': sdf,
            'surrogate_sdf': sdf,
            'surrogate_reinitialized_sdf': sdf,
            'diffusion_gaussian_0_5': sdf * 0.9,
            'diffusion_gaussian_1_0': sdf * 0.8,
            'diffusion_gaussian_2_0': sdf * 0.7,
            'diffusion_gaussian_3_0': sdf * 0.6,
        }, window_path)
        window_paths.append(window_path)
    result = save_diffusion_gif(
        window_paths, layout, 0, tmp_path / 'comparison.gif', max_frames=3,
        variant=variant,
    )
    with Image.open(result) as animation:
        assert animation.format == 'GIF'
        assert animation.n_frames == 3
        assert animation.info['loop'] == 0
