from dataclasses import dataclass, fields, replace
from typing import Callable, Dict, List, Optional, Tuple
import json
import random

import numpy as np
import h5py as h5
import torch
from torch.utils.data import Dataset

from nucleus.data.batching import Data, make_data
from nucleus.data.normalize import Normalizer
from nucleus.physics.poisson import helmholtz_from_faces, GRID_SPACING


@dataclass
class DivFreeData:
    """Divergence-free fields, each on its natural grid.

    Used for both the input and the target of the divergence-free forecast
    dataset; a collate function (added later) batches these across samples.

    Shapes (leading time dim ``T`` per dataset sample; the batch dim is added by
    the collate):
        ``sdf``, ``temperature``, ``phi``: ``(T, H, W)`` cell-centered.
        ``velx``: ``(T, H, W+1)`` x-face velocity.
        ``vely``: ``(T, H+1, W)`` y-face velocity.
        ``psi``: ``(T, H+1, W+1)`` nodal streamfunction.
    """
    sdf: torch.Tensor
    temperature: torch.Tensor
    velx: torch.Tensor
    vely: torch.Tensor
    psi: torch.Tensor
    phi: torch.Tensor

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "DivFreeData":
        """Return a new DivFreeData with ``fn`` applied to every field."""
        return replace(self, **{f.name: fn(getattr(self, f.name)) for f in fields(self)})

    @classmethod
    def stack(cls, items: List["DivFreeData"]) -> "DivFreeData":
        """Stack a list of per-sample DivFreeData into a batched one, adding a new
        leading batch dim to every field (e.g. ``sdf`` becomes ``(B, T, H, W)``)."""
        return cls(**{
            field.name: torch.stack([getattr(item, field.name) for item in items])
            for field in fields(cls)
        })

    def to(self, device: torch.device, non_blocking: bool = False) -> "DivFreeData":
        return self._apply(lambda tensor: tensor.to(device, non_blocking=non_blocking))

    def detach(self) -> "DivFreeData":
        return self._apply(lambda tensor: tensor.detach())

    def pin_memory(self) -> "DivFreeData":
        return self._apply(lambda tensor: tensor.pin_memory())


@dataclass
class DivFreeBatch:
    input: DivFreeData
    target: Optional[DivFreeData]
    sim_params: List[Dict]
    dx: torch.Tensor
    dy: torch.Tensor
    sim_params_tensor: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "DivFreeBatch":
        move = lambda tensor: tensor.to(device, non_blocking=non_blocking) if tensor is not None else None
        return DivFreeBatch(
            input=self.input.to(device, non_blocking=non_blocking),
            target=self.target.to(device, non_blocking=non_blocking) if self.target is not None else None,
            sim_params=self.sim_params,
            dx=move(self.dx),
            dy=move(self.dy),
            sim_params_tensor=move(self.sim_params_tensor),
        )

    def detach(self) -> "DivFreeBatch":
        detach = lambda tensor: tensor.detach() if tensor is not None else None
        return DivFreeBatch(
            input=self.input.detach(),
            target=self.target.detach() if self.target is not None else None,
            sim_params=self.sim_params,
            dx=detach(self.dx),
            dy=detach(self.dy),
            sim_params_tensor=detach(self.sim_params_tensor),
        )

    def pin_memory(self) -> "DivFreeBatch":
        pin = lambda tensor: tensor.pin_memory() if tensor is not None else None
        return DivFreeBatch(
            input=self.input.pin_memory(),
            target=self.target.pin_memory() if self.target is not None else None,
            sim_params=self.sim_params,
            dx=pin(self.dx),
            dy=pin(self.dy),
            sim_params_tensor=pin(self.sim_params_tensor),
        )


def _grid_spacing(sim_params: Dict) -> Tuple[float, float]:
    dx = (sim_params["x_max"] - sim_params["x_min"]) / (
        sim_params["num_blocks_x"] * int(sim_params["nx_block"])
    )
    dy = (sim_params["y_max"] - sim_params["y_min"]) / (
        sim_params["num_blocks_y"] * int(sim_params["ny_block"])
    )
    return dx, dy


def divfree_collate(
    batch: List[Tuple[DivFreeData, DivFreeData, Dict, torch.Tensor]]
) -> DivFreeBatch:
    inputs, targets, sim_params, sim_params_tensors = zip(*batch)
    spacings = [_grid_spacing(sample_params) for sample_params in sim_params]
    return DivFreeBatch(
        input=DivFreeData.stack(list(inputs)),
        target=DivFreeData.stack(list(targets)),
        sim_params=list(sim_params),
        dx=torch.tensor([dx for dx, _ in spacings]),
        dy=torch.tensor([dy for _, dy in spacings]),
        sim_params_tensor=torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in sim_params_tensors]),
    )

# Fields read from disk. The streamfunction psi and potential phi are computed
# from the face velocities; every field is then kept on its natural grid (see
# DivFreeData): dfun/temperature/phi cell-centered, velfacex/velfacey on the
# staggered faces, psi on the nodes.
BASE_FIELDS = ["dfun", "temperature", "velfacex", "velfacey"]


class InMemDivFreeForecastDataset(Dataset):
    """In-memory forecast dataset for the divergence-free model.

    Loads the face-staggered velocity (``velfacex``, ``velfacey``) alongside
    ``dfun`` and ``temperature``, computes the streamfunction ``psi`` (nodal) and
    the potential ``phi`` (cell-centered) from the exact staggered operators, and
    returns each field on its natural grid as a :class:`DivFreeData` -- no
    cell-splitting or channel stacking. The input and target windows are each a
    ``DivFreeData``.
    """

    def __init__(
        self,
        filenames: List[str],
        input_fields: Optional[List[str]],
        output_fields: Optional[List[str]],
        future_time_window: int,
        history_time_window: int,
        time_step: int,
        start_time: int,
        fluid_params: List[str],
        heater_params: List[str],
        global_params: List[str],
        layout: str,
        normalizer: Optional[Normalizer],
        augment: bool = False,
    ):
        super().__init__()
        # input_fields / output_fields are accepted for a common constructor
        # signature but the field set is fixed (BASE_FIELDS + psi + phi).
        self.filenames = filenames
        self.future_time_window = future_time_window
        self.history_time_window = history_time_window
        self.time_step = time_step
        self.start_time = start_time
        self.fluid_params = fluid_params
        self.heater_params = heater_params
        self.global_params = global_params
        self.layout = layout
        self.normalizer = normalizer
        self.augment = augment

        self.data = self._load_data()
        self.traj_lens = [trajectory[BASE_FIELDS[0]].shape[0] for trajectory in self.data]

        self.sim_params = []
        for filename in filenames:
            with open(filename.replace(".hdf5", ".json"), "r", encoding="utf-8") as handle:
                self.sim_params.append(json.load(handle))

    def _load_data(self) -> List[Dict[str, torch.Tensor]]:
        data = []
        for filename in self.filenames:
            with h5.File(filename, "r") as handle:
                data.append({field: torch.tensor(handle[field][...]) for field in BASE_FIELDS})
        return data

    def _get_traj_len(self, traj_len: int) -> int:
        return traj_len - self.start_time - self.future_time_window - self.history_time_window + 1

    def __len__(self) -> int:
        return sum(self._get_traj_len(traj_len) for traj_len in self.traj_lens)

    def _flip_width(self, fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Mirror across the vertical center line: flip along width and negate the
        # x-face velocity (its direction reverses).
        flipped = {name: torch.flip(tensor, dims=[-1]) for name, tensor in fields.items()}
        flipped["velfacex"] = -flipped["velfacex"]
        return flipped

    def _to_divfree_data(self, fields: Dict[str, torch.Tensor], bulk_temp: int) -> DivFreeData:
        # Compute psi/phi from the exact staggered faces and normalize every field
        # on its natural grid. Nothing is split onto cells: each field keeps its own
        # (cell / face / nodal) shape.
        sdf, temperature = fields["dfun"], fields["temperature"]
        velfacex, velfacey = fields["velfacex"], fields["velfacey"]
        # psi (T, H+1, W+1) nodal, phi (T, H, W) cell-centered
        psi, phi = helmholtz_from_faces(velfacex, velfacey, GRID_SPACING, GRID_SPACING)

        if self.normalizer is not None:
            normalizer = self.normalizer
            sdf = normalizer.normalize_sdf(sdf)
            temperature = normalizer.normalize_temp(temperature, bulk_temp)
            velfacex = normalizer.normalize_velx(velfacex)
            velfacey = normalizer.normalize_vely(velfacey)
            psi = normalizer.normalize_psi(psi)
            phi = normalizer.normalize_phi(phi)

        return DivFreeData(
            sdf=sdf.float(),
            temperature=temperature.float(),
            velx=velfacex.float(),
            vely=velfacey.float(),
            psi=psi.float(),
            phi=phi.float(),
        )

    def __getitem__(self, idx: int) -> Data:
        cumulative_samples = np.cumsum([self._get_traj_len(traj_len) for traj_len in self.traj_lens])
        file_idx = np.searchsorted(cumulative_samples, idx, side="right")
        start = idx + self.start_time - (cumulative_samples[file_idx - 1] if file_idx > 0 else 0)

        inp_slice = slice(start, start + self.history_time_window, self.time_step)
        out_slice = slice(
            start + self.history_time_window,
            start + self.history_time_window + self.future_time_window,
            self.time_step,
        )

        inp_fields = {name: self.data[file_idx][name][inp_slice] for name in BASE_FIELDS}
        out_fields = {name: self.data[file_idx][name][out_slice] for name in BASE_FIELDS}

        if self.augment and random.random() < 0.5:
            inp_fields = self._flip_width(inp_fields)
            out_fields = self._flip_width(out_fields)

        sim_params_physical = self.sim_params[file_idx]
        bulk_temp = int(sim_params_physical["bulk_temp"])

        # Input and target are each a DivFreeData (fields on their natural grids).
        # make_data still carries the grids and sim-param tensor; a DivFreeData-aware
        # collate (added later) will batch the fields.
        inp_data = self._to_divfree_data(inp_fields, bulk_temp)
        out_data = self._to_divfree_data(out_fields, bulk_temp)

        # sim_params_tensor (the network's conditioning input) needs normalized
        # values, but DivFreeBatch.sim_params -- consumed by step()'s temperature
        # ansatz / continuity as physical constants (sat_temp, heater wallTemp/xMin/
        # xMax, ...) -- must stay physical, the same units run_forward_trajectory.py
        # loads directly from the JSON at rollout time. Normalizing the dict that
        # became DivFreeBatch.sim_params here used to silently mismatch those units
        # between training and rollout.
        sim_params_for_tensor = sim_params_physical
        if self.normalizer is not None:
            sim_params_for_tensor = self.normalizer.normalize_params([sim_params_physical])[0]

        sim_params_tensor = np.array(
            [sim_params_for_tensor[p] for p in self.fluid_params] +
            [sim_params_for_tensor["heater"][p] for p in self.heater_params] +
            [sim_params_for_tensor[p] for p in self.global_params]
        )

        return (
            inp_data,
            out_data,
            sim_params_physical,
            sim_params_tensor
        )