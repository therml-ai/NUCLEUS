from nucleus.models import get_model
from nucleus.data.batching import CollatedBatch
from nucleus.testing.parametrize import parametrize_available_devices
import torch

@parametrize_available_devices("device")
def test_bubbleformer_vit(device):
    model = get_model(
        "bubbleformer_vit",
        input_fields=4,
        output_fields=4,
        time_window=8,
        patch_size=4,
        embed_dim=256,
        num_heads=4,
        processor_blocks=12,
        attn_scale=True,
        feat_scale=True,
    )
    model = model.to(device)

    # bubbleformer uses a [B, T, C, H, W] layout
    batch = CollatedBatch(
        input=torch.randn(4, 8, 4, 64, 64),
        target=None,
        fluid_params_dict={},
        x_grid=torch.randn(64),
        y_grid=torch.randn(64),
        dx=torch.tensor(0.01),
        dy=torch.tensor(0.01),
        fluid_params_tensor=torch.randn(4, 16),
    ).to(device)
    
    output = model(batch)
    assert output.shape == (4, 8, 4, 64, 64)
    assert torch.all(torch.isfinite(output))
    
@parametrize_available_devices("device")
def test_bubbleformer_film_vit(device):
    model = get_model(
        "bubbleformer_film_vit",
        input_fields=4,
        output_fields=4,
        time_window=8,
        patch_size=4,
        embed_dim=256,
        num_heads=4,
        processor_blocks=12,
        attn_scale=True,
        feat_scale=True,
        num_fluid_params=16,
    )
    model = model.to(device)

    # bubbleformer uses a [B, T, C, H, W] layout
    batch = CollatedBatch(
        input=torch.randn(4, 8, 4, 64, 64),
        target=None,
        fluid_params_dict={},
        x_grid=torch.randn(64),
        y_grid=torch.randn(64),
        dx=torch.tensor(0.01),
        dy=torch.tensor(0.01),
        fluid_params_tensor=torch.randn(4, 16),
    ).to(device)
    
    output = model(batch)
    assert output.shape == (4, 8, 4, 64, 64)
    assert torch.all(torch.isfinite(output))