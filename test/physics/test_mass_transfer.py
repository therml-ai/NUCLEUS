import torch

from nucleus.physics.mass_transfer import interface_heatflux, mass_transfer
from nucleus.physics.sdf import interface_mask


def _vertical_interface(height, width, dx, dy, sat_temp, liquid_slope, vapor_slope):
    """A vertical interface with liquid (sdf < 0) on the left and vapor (sdf >= 0)
    on the right, each phase linear in x with its own slope and meeting sat_temp at
    the interface. The normal is +x, so grad(T).n equals the phase's slope."""
    x = (torch.arange(width, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(height, dtype=torch.float64) + 0.5) * dy
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    x0 = x[width // 2] + 0.5 * dx                      # off a cell center -> no sdf == 0
    sdf = grid_x - x0
    temp = torch.where(
        sdf >= 0,
        sat_temp + vapor_slope * (grid_x - x0),
        sat_temp + liquid_slope * (grid_x - x0),
    )
    return temp, sdf


def test_heatflux_sides_overlap_on_the_band():
    # Regression: the liquid and vapor one-sided gradients must be defined on the
    # SAME interface cells. If they have disjoint support, mass_transfer never
    # subtracts them and the flux collapses to a single side (max exactly zero).
    H = W = 48
    dx = dy = 1.0 / 32
    temp, sdf = _vertical_interface(H, W, dx, dy, sat_temp=1.0, liquid_slope=2.0, vapor_slope=10.0)

    liquid_side, vapor_side = interface_heatflux(temp, sdf, 1.0, dx, dy)
    band = interface_mask(sdf)

    both_nonzero = (liquid_side != 0) & (vapor_side != 0) & band
    assert both_nonzero.any(), "liquid and vapor heat fluxes never overlap on the band"
    assert torch.all((liquid_side != 0)[band]), "liquid side missing on some band cells"
    assert torch.all((vapor_side != 0)[band]), "vapor side missing on some band cells"


def test_mass_transfer_can_be_either_sign():
    # The jump must be able to change sign: evaporation (negative) and condensation
    # (positive) depending on which side conducts more heat. The pre-fix code could
    # only ever produce one sign.
    H = W = 48
    dx = dy = 1.0 / 32
    stefan, reynolds, prandtl, vapor_conductivity = 1.0, 1.0, 1.0, 0.5

    # Liquid conducts more -> positive (condensation).
    temp, sdf = _vertical_interface(H, W, dx, dy, 1.0, liquid_slope=10.0, vapor_slope=2.0)
    mdot_positive = mass_transfer(
        temp, sdf, 1.0, dx, dy,
        stefan=stefan, reynolds=reynolds, prandtl=prandtl, thermal_conductivity=vapor_conductivity,
    )
    assert (mdot_positive > 0).any()

    # Vapor conducts more -> negative (evaporation).
    temp, sdf = _vertical_interface(H, W, dx, dy, 1.0, liquid_slope=2.0, vapor_slope=10.0)
    mdot_negative = mass_transfer(
        temp, sdf, 1.0, dx, dy,
        stefan=stefan, reynolds=reynolds, prandtl=prandtl, thermal_conductivity=vapor_conductivity,
    )
    assert (mdot_negative < 0).any()


def test_mass_transfer_batches_over_leading_dims():
    H = W = 32
    dx = dy = 1.0 / 32
    temp, sdf = _vertical_interface(H, W, dx, dy, 1.0, 2.0, 10.0)
    batch_temp = temp.expand(2, 3, H, W)
    batch_sdf = sdf.expand(2, 3, H, W)

    mdot = mass_transfer(
        batch_temp, batch_sdf, 1.0, dx, dy,
        stefan=1.0, reynolds=1.0, prandtl=1.0, thermal_conductivity=0.5,
    )
    assert mdot.shape == (2, 3, H, W)
    single = mass_transfer(
        temp, sdf, 1.0, dx, dy,
        stefan=1.0, reynolds=1.0, prandtl=1.0, thermal_conductivity=0.5,
    )
    assert torch.allclose(mdot[1, 2], single)
