import math

import torch

from nucleus.physics.guard_cells import (
    add_halo,
    add_level_set_halo,
    add_temperature_halo,
    crop_halo,
    refill_guard_cells,
)


def test_add_halo_replicates_edges_and_crop_inverts_it():
    field = torch.arange(12, dtype=torch.float64).reshape(3, 4).expand(2, 3, 4)
    padded = add_halo(field, 2)
    assert padded.shape == (2, 7, 8)
    assert torch.equal(padded[:, 0, 2:-2], field[:, 0])
    assert padded[1, 0, 0] == field[1, 0, 0] and padded[1, -1, -1] == field[1, -1, -1]
    assert torch.equal(crop_halo(padded, 2), field)
    assert add_halo(field, 0) is field


def test_temperature_halo_reflects_about_the_wall():
    height, width, halo = 5, 3, 2
    temp = torch.arange(height, dtype=torch.float64).reshape(-1, 1).expand(height, width)
    wall_temp = 10.0

    padded = add_temperature_halo(temp, halo, wall_temp)
    # ghost row halo-1 mirrors interior row 0, ghost row halo-2 mirrors row 1
    assert torch.allclose(padded[halo - 1, halo:-halo], 2 * wall_temp - temp[0])
    assert torch.allclose(padded[halo - 2, halo:-halo], 2 * wall_temp - temp[1])
    assert torch.equal(padded[halo:, halo:-halo], add_halo(temp, halo)[halo:, halo:-halo])
    assert torch.equal(add_temperature_halo(temp, halo, None), add_halo(temp, halo))


def test_level_set_halo_extends_at_the_contact_angle():
    height, width, halo, dy = 4, 3, 3, 0.25
    sdf = torch.full((height, width), 1.0, dtype=torch.float64)

    square = add_level_set_halo(sdf, halo, dy, None)
    assert torch.all(square[:halo] == 1.0)

    angled = add_level_set_halo(sdf, halo, dy, 60.0)
    step = dy * math.cos(math.radians(60.0))
    for ghost_row in range(halo):
        expected = 1.0 - (halo - ghost_row) * step
        assert torch.allclose(angled[ghost_row], torch.full((width + 2 * halo,), expected, dtype=torch.float64))


def test_level_set_halo_broadcasts_batched_contact_angle():
    sdf = torch.ones(2, 1, 4, 3, dtype=torch.float64)
    angles = torch.tensor([90.0, 0.0], dtype=torch.float64)[:, None, None, None]

    padded = add_level_set_halo(sdf, 2, 0.5, angles)
    assert torch.allclose(padded[0, 0, :2], torch.ones(2, 7, dtype=torch.float64))
    assert torch.allclose(padded[1, 0, 1], torch.full((7,), 0.5, dtype=torch.float64))
    assert torch.allclose(padded[1, 0, 0], torch.full((7,), 0.0, dtype=torch.float64))


def test_refill_guard_cells_overwrites_halo_with_interior_values():
    field = torch.arange(36, dtype=torch.float64).reshape(6, 6)
    out = refill_guard_cells(field, 2)

    assert torch.equal(out[2:-2, 2:-2], field[2:-2, 2:-2])
    for row in (0, 1):
        assert torch.equal(out[row, 2:-2], field[2, 2:-2])
    assert out[0, 0] == field[2, 2] and out[-1, -1] == field[-3, -3]
    assert refill_guard_cells(field, 0) is field
