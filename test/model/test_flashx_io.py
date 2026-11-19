import pytest
import torch

from nucleus.models.flashx_io import next_frames, num_cell_channels
from nucleus.testing.flashx import random_batch


def _prediction(num_timesteps):
    """A prediction whose dfun is its frame index everywhere."""
    batch = random_batch(["dfun"], {}, batch_size=1, timesteps=num_timesteps, height=4, width=4)
    frames = torch.arange(num_timesteps, dtype=torch.float32)[None, :, None, None]
    return batch.with_fields({"dfun": frames.expand(1, -1, 4, 4)})


@pytest.mark.parametrize(("num_timesteps", "target_timesteps", "frames"), [
    (8, 8, list(range(8))),
    (2, 8, [0, 1]),
    (2, 4, [4, 5]),
    (1, 1, [7]),
])
def test_next_frames_are_the_first_of_the_target_window(num_timesteps, target_timesteps, frames):
    # the model's 8 output frames line up with the target by their last target_timesteps
    selected = next_frames(_prediction(8), num_timesteps, target_timesteps)
    assert selected.fields["dfun"][0, :, 0, 0].tolist() == [float(frame) for frame in frames]


@pytest.mark.parametrize(("num_timesteps", "target_timesteps"), [(3, 2), (0, 2), (2, 9)])
def test_next_frames_rejects_windows_the_model_does_not_predict(num_timesteps, target_timesteps):
    with pytest.raises(ValueError, match="num_timesteps <= target_timesteps"):
        next_frames(_prediction(8), num_timesteps, target_timesteps)


def test_a_face_field_takes_two_cell_channels_and_a_cell_field_one():
    assert num_cell_channels(["dfun", "temperature"]) == 2
    assert num_cell_channels(["dfun", "temperature", "velfacex", "velfacey"]) == 6
    batch = random_batch(["dfun", "velfacex", "velfacey"], {}, batch_size=1)
    assert batch.stack_field_cells(["dfun", "velfacex", "velfacey"]).shape[-1] == num_cell_channels(
        ["dfun", "velfacex", "velfacey"]
    )
