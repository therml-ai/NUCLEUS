import random

import pytest
import torch
from boiling_data.tasks.forecast.flashx_batch import FlashXForecastBatch

from nucleus.testing.flashx import random_batch
from nucleus.training.push_forward import push_forward, sample_num_unrolls

CONFIG_KEYS = {"physical": ["bulk_temp"]}


def _frames(first_frame, num_timesteps):
    """A batch whose dfun is the frame index everywhere."""
    batch = random_batch(["dfun"], CONFIG_KEYS, timesteps=num_timesteps, height=4, width=4)
    frames = torch.arange(first_frame, first_frame + num_timesteps, dtype=torch.float32)
    return batch.with_fields({"dfun": frames[None, :, None, None].expand(2, -1, 4, 4)})


def _frame_indices(batch):
    return batch.fields["dfun"][0, :, 0, 0].tolist()


def _predict_next_two_frames(window):
    return window.with_fields({"dfun": window.fields["dfun"][:, -2:] + 2})


def test_push_forward_unrolls_the_input_and_picks_the_matching_target():
    batch = FlashXForecastBatch(_frames(0, 3), _frames(3, 6))
    pushed = push_forward(_predict_next_two_frames, batch, num_unrolls=2, target_timesteps=2)
    assert _frame_indices(pushed.input) == [4.0, 5.0, 6.0]
    assert _frame_indices(pushed.target) == [7.0, 8.0]


def test_no_unrolls_keeps_the_input_and_the_first_target_window():
    batch = FlashXForecastBatch(_frames(0, 3), _frames(3, 6))
    pushed = push_forward(_predict_next_two_frames, batch, num_unrolls=0, target_timesteps=2)
    assert _frame_indices(pushed.input) == [0.0, 1.0, 2.0]
    assert _frame_indices(pushed.target) == [3.0, 4.0]


def test_sample_num_unrolls_respects_the_probability_and_bound():
    generator = random.Random(0)
    assert {sample_num_unrolls(3, 0.0, generator) for _ in range(50)} == {0}
    assert {sample_num_unrolls(3, 1.0, generator) for _ in range(200)} == {1, 2, 3}
    assert sample_num_unrolls(0, 1.0, generator) == 0
    with pytest.raises(ValueError):
        sample_num_unrolls(1, 1.5)
