from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Trajectory:
    """A rollout trajectory -- the input container for a model's forward_trajectory.

    Each field carries a batch and time axis, ``(B, T, ...)``. The spatial grids
    are whatever the target model expects: collocated (every field on the same
    ``(H, W)`` grid) for Nucleus2MoE, or the natural staggered grids for
    Nucleus2MoEDivFree (``velx`` on ``(H, W+1)`` faces, ``vely`` on ``(H+1, W)``).
    Converting between grids is the responsibility of whatever builds the
    Trajectory, since it depends on how the data was generated -- it is
    deliberately not done here.

    ``sim_params`` holds one physical sim-parameter dict per batch element.
    """
    sdf: torch.Tensor
    temp: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor
    sim_params: List[dict]

    @property
    def num_steps(self) -> int:
        return self.sdf.shape[1]

    def first(self, num_steps: int) -> "Trajectory":
        return self._select_steps(slice(None, num_steps))

    def last(self, num_steps: int) -> "Trajectory":
        return self._select_steps(slice(-num_steps, None))

    def _select_steps(self, steps: slice) -> "Trajectory":
        return Trajectory(
            sdf=self.sdf[:, steps],
            temp=self.temp[:, steps],
            velx=self.velx[:, steps],
            vely=self.vely[:, steps],
            sim_params=self.sim_params,
        )

    def extend(
        self,
        sdf: torch.Tensor,
        temp: torch.Tensor,
        velx: torch.Tensor,
        vely: torch.Tensor,
    ) -> "Trajectory":
        return Trajectory(
            sdf=torch.cat((self.sdf, sdf), dim=1),
            temp=torch.cat((self.temp, temp), dim=1),
            velx=torch.cat((self.velx, velx), dim=1),
            vely=torch.cat((self.vely, vely), dim=1),
            sim_params=self.sim_params,
        )


@dataclass(frozen=True)
class RolloutWindow:
    """The frames of a case that seed a rollout, and the frames it is compared to."""
    start_time: int
    history_steps: int
    trajectory_steps: int
    # the window sizes and grid spacing rollouts were always run with
    input_window_size: int = 8
    output_window_size: int = 8
    dx: float = 1 / 32

    @property
    def history(self) -> slice:
        return slice(self.start_time, self.start_time + self.history_steps)

    @property
    def targets(self) -> slice:
        return slice(self.start_time, self.start_time + self.trajectory_steps)
