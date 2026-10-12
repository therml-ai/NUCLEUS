import pytest
import torch

from nucleus.testing.flashx import random_forecast_batch
from nucleus.training.field_dropout import FieldDropout

CONFIG_KEYS = {"physical": ["bulk_temp"]}


def _batch():
    return random_forecast_batch(["dfun", "velfacex"], CONFIG_KEYS, batch_size=64, height=4, width=4)


def test_whole_fields_of_samples_are_zeroed_and_the_target_kept():
    batch = _batch()
    dropped = FieldDropout(drop_probability=0.5, generator=torch.Generator().manual_seed(0))(batch)
    assert dropped.target is batch.target
    for name, tensor in dropped.input.fields.items():
        zeroed = (tensor == 0).flatten(start_dim=1).all(dim=1)
        kept = torch.isclose(tensor, batch.input.fields[name]).flatten(start_dim=1).all(dim=1)
        assert torch.all(zeroed | kept)
        assert 0 < zeroed.sum() < 64


def test_eval_mode_and_zero_probability_change_nothing():
    batch = _batch()
    assert FieldDropout(0.5).eval()(batch) is batch
    unchanged = FieldDropout(0.0)(batch)
    for name, tensor in batch.input.fields.items():
        torch.testing.assert_close(unchanged.input.fields[name], tensor)


def test_drop_probability_must_be_a_probability():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FieldDropout(1.5)
