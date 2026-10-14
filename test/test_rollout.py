import numpy as np
import pytest
import torch
from boiling_data.bubbleml import read_bubbleml
from boiling_data.flashx.batch import simulation_from_batch

from nucleus.models import get_model
from nucleus.models.nucleus2_moe_divfree import Nucleus2MoEDivFree
from nucleus.rollout import RolloutSettings, rollout, rollout_case, save_rollout
from nucleus.testing.flashx import flashx_normalizer, random_batch

FIELD_NAMES = Nucleus2MoEDivFree.field_names


class _NextFrameCounter(torch.nn.Module):
    """Predicts the frames after a window as the last frame's index plus one, two, ..."""
    field_names = ("dfun",)

    def predict_next(self, window, num_timesteps, target_timesteps):
        last = window.fields["dfun"][:, -1:]
        steps = torch.arange(1, num_timesteps + 1, dtype=last.dtype)[None, :, None, None]
        return window.with_fields({"dfun": last + steps})


def _frames(num_timesteps):
    batch = random_batch(["dfun"], {}, batch_size=1, timesteps=num_timesteps, height=4, width=4)
    frames = torch.arange(num_timesteps, dtype=torch.float32)[None, :, None, None]
    return batch.with_fields({"dfun": frames.expand(1, -1, 4, 4)})


@pytest.mark.parametrize("num_steps", [1, 5, 8])
def test_rollout_predicts_exactly_num_steps_frames_in_order(num_steps):
    predicted = rollout(_NextFrameCounter(), _frames(3), num_steps, frames_per_step=2, target_timesteps=3)
    assert predicted.fields["dfun"][0, :, 0, 0].tolist() == [float(frame) for frame in range(3, 3 + num_steps)]


def test_rollout_checks_its_settings():
    with pytest.raises(ValueError, match="frames_per_step"):
        rollout(_NextFrameCounter(), _frames(3), 4, frames_per_step=3, target_timesteps=2)
    with pytest.raises(ValueError, match="target_timesteps"):
        rollout(_NextFrameCounter(), _frames(3), 4, frames_per_step=1, target_timesteps=4)
    with pytest.raises(ValueError, match="num_steps"):
        rollout(_NextFrameCounter(), _frames(3), 0, frames_per_step=1, target_timesteps=1)


@pytest.fixture
def model():
    torch.manual_seed(0)
    normalizer = flashx_normalizer(FIELD_NAMES, vector_fields=Nucleus2MoEDivFree.vector_fields)
    return get_model(
        "nucleus2_moe_divfree", normalizer, patch_size=4, embed_dim=64, num_heads=2, processor_blocks=1,
        num_experts=2, topk=1, moe_intermediate_dim=32, patching="Linear",
    ).eval()


@pytest.fixture
def case_path(tmp_path):
    """A BubbleML case of six frames, half a time unit apart."""
    batch = random_batch(FIELD_NAMES, Nucleus2MoEDivFree.config_keys, batch_size=1, timesteps=6)
    return simulation_from_batch(batch, 0.5 * np.arange(6)).to_bubbleml(tmp_path / "case.hdf5")


def test_a_rollout_is_written_as_a_bubbleml_file(model, case_path, tmp_path):
    settings = RolloutSettings(start_frame=2, history_timesteps=2, trajectory_steps=3, target_timesteps=2, frames_per_step=2)
    prediction, time = rollout_case(model, case_path, settings, torch.device("cpu"))
    np.testing.assert_allclose(time, [2.0, 2.5, 3.0])

    written = read_bubbleml(save_rollout(prediction, time, tmp_path / "rollouts" / "case.hdf5"))
    source = read_bubbleml(case_path)
    np.testing.assert_allclose(written.time, time)
    assert written.parameters == source.parameters
    for name in FIELD_NAMES:
        assert written.field(name).data.shape == (3, *source.field(name).data.shape[1:])
        np.testing.assert_allclose(written.field(name).grid_x, source.field(name).grid_x)
        np.testing.assert_allclose(
            written.field(name).data, prediction.fields[name][0].double().numpy(), rtol=1e-6
        )


def test_a_case_too_short_for_the_history_is_rejected(model, case_path):
    settings = RolloutSettings(start_frame=5, history_timesteps=2, trajectory_steps=1, target_timesteps=1, frames_per_step=1)
    with pytest.raises(ValueError, match="no 2 frames from frame 5"):
        rollout_case(model, case_path, settings, torch.device("cpu"))
