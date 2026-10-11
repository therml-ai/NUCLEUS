from types import SimpleNamespace

import pytest
import torch

from nucleus.run_forward_trajectory import run_test
from nucleus.trajectory import RolloutWindow, Trajectory


def test_rollout_window_slices():
    window = RolloutWindow(start_time=10, history_steps=8, trajectory_steps=30)
    assert window.history == slice(10, 18)
    assert window.targets == slice(10, 40)


def _counting_trajectory(num_steps: int) -> Trajectory:
    steps = torch.arange(num_steps, dtype=torch.float32).reshape(1, num_steps, 1, 1)
    return Trajectory(sdf=steps, temp=steps, velx=steps, vely=steps, sim_params=[{}])


def test_trajectory_first_and_last_keep_the_ends():
    trajectory = _counting_trajectory(num_steps=10)
    assert trajectory.first(3).sdf.flatten().tolist() == [0, 1, 2]
    assert trajectory.last(3).velx.flatten().tolist() == [7, 8, 9]


def test_run_test_rejects_a_model_without_rollout():
    model_without_rollout = torch.nn.Linear(1, 1)
    config = SimpleNamespace(start_time=0, history_time_window=8)
    with pytest.raises(TypeError, match="Linear does not implement rollout"):
        run_test(config, model_without_rollout, normalizer=None, test_file_path="unused.hdf5", trajectory_steps=8)
