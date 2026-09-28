"""Visualize the velocity-construction utilities directly, without the model.

Example:
    python scripts/visualize_divergence.py --show
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nucleus.models.nucleus2_moe_divfree import (
    velocity_from_potentials,
    DOMAIN_X_MIN,
    DOMAIN_X_MAX,
    DOMAIN_Y_MIN,
    DOMAIN_Y_MAX,
    _cell_centers,
)
from nucleus.physics.sdf import band_mask


def domain_grid(height, width):
    y = _cell_centers(height, DOMAIN_Y_MIN, DOMAIN_Y_MAX, "cpu", torch.float32)
    x = _cell_centers(width, DOMAIN_X_MIN, DOMAIN_X_MAX, "cpu", torch.float32)
    return torch.meshgrid(y, x, indexing="ij")   # grid_y, grid_x


def circle_sdf(height, width, radius, center_x, center_y):
    """Signed distance to a circle: positive (vapor) inside, negative (liquid)
    outside -- matching the model's dfun convention (sdf > 0 == vapor)."""
    grid_y, grid_x = domain_grid(height, width)
    distance = torch.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)
    return radius - distance


def smooth_random_field(height, width, num_modes, generator):
    """An arbitrary but smooth scalar potential: a random sum of low-frequency
    Fourier modes over the domain. Smoothness keeps the derivative-based velocity
    legible (white-noise potentials would give white-noise velocity)."""
    grid_y, grid_x = domain_grid(height, width)
    u = (grid_x - DOMAIN_X_MIN) / (DOMAIN_X_MAX - DOMAIN_X_MIN)
    v = (grid_y - DOMAIN_Y_MIN) / (DOMAIN_Y_MAX - DOMAIN_Y_MIN)

    field = torch.zeros(height, width)
    for _ in range(num_modes):
        freq_x = torch.randint(1, 4, (1,), generator=generator).item()
        freq_y = torch.randint(1, 4, (1,), generator=generator).item()
        amplitude = torch.randn(1, generator=generator).item()
        phase = 2 * torch.pi * torch.rand(1, generator=generator).item()
        field = field + amplitude * torch.sin(2 * torch.pi * (freq_x * u + freq_y * v) + phase)
    return field / num_modes**0.5


def wall_vanishing_envelope(height, width, length_scale=2.0):
    """Smooth envelope that goes to zero at the closed walls on the *physical*
    scale (length_scale in domain units). Multiplying a random psi by it mimics a
    real streamfunction, which is const == 0 on a wall (no-penetration). The top
    is open (envelope -> 1)."""
    grid_y, grid_x = domain_grid(height, width)
    return (torch.tanh((grid_y - DOMAIN_Y_MIN) / length_scale)
            * torch.tanh((grid_x - DOMAIN_X_MIN) / length_scale)
            * torch.tanh((DOMAIN_X_MAX - grid_x) / length_scale))


def pointwise_divergence(velx, vely, dx, dy):
    """Central-difference divergence field (same stencil as
    physical_metrics.divergence, kept per-cell instead of averaged)."""
    (velx_grad_x,) = torch.gradient(velx, spacing=dx, dim=-1)
    (vely_grad_y,) = torch.gradient(vely, spacing=dy, dim=-2)
    return velx_grad_x + vely_grad_y


def liquid_interior_mask(sdf, band, radius, wall_cells):
    """Liquid cells (gate == 0) at least `radius` cells from any vapor/interface
    cell and `wall_cells` from the border -- where the velocity is exactly
    div-free by construction."""
    near_interface_or_vapor = (sdf > -band).float()[None, None]
    dilated = F.max_pool2d(near_interface_or_vapor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    interior = dilated[0, 0] == 0
    interior[:wall_cells] = False
    interior[-wall_cells:] = False
    interior[:, :wall_cells] = False
    interior[:, -wall_cells:] = False
    return interior


def check_boundary_conditions(velx, vely, divergence, sdf, band, div_tolerance, slip_tolerance):
    """Check the boundary conditions the construction guarantees:

      - divergence free in the contiguous liquid interior,
      - zero divergence in the first cell against every closed wall
        (both measured relative to the interface's peak |div|, so scale-free), and
      - free-slip: the wall-normal velocity is small relative to the interior speed
        (v . n -> 0 at the closed walls) while the tangential velocity stays free.

    Prints each with PASS/FAIL. Returns True iff every check passes.
    """
    peak = divergence.abs().max().item()
    scale = max(peak, 1e-8)

    wall_band = torch.cat([divergence[0, :], divergence[:, 0], divergence[:, -1]])
    div_checks = [("zero divergence at closed walls", wall_band.abs().max().item())]
    liquid_interior = liquid_interior_mask(sdf, band=band, radius=3, wall_cells=4)
    if liquid_interior.any():
        div_checks.insert(0, ("divergence free in liquid interior",
                              divergence[liquid_interior].abs().max().item()))

    print(f"boundary-condition checks:")
    all_pass = True
    for name, value in div_checks:
        ratio = value / scale
        passed = ratio < div_tolerance
        all_pass = all_pass and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<34} max|div| = {value:.2e}  "
              f"({ratio:.1e} of peak, tol {div_tolerance:.0e})")

    # Free-slip: wall-normal velocity small relative to the interior speed.
    wall_normal = torch.cat([vely[0, :].abs(), velx[:, 0].abs(), velx[:, -1].abs()]).mean().item()
    interior_speed = torch.sqrt(velx**2 + vely**2)[4:-4, 4:-4].mean().item()
    normal_ratio = wall_normal / max(interior_speed, 1e-8)
    passed = normal_ratio < slip_tolerance
    all_pass = all_pass and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {'free-slip (v.n -> 0 at walls)':<34} "
          f"mean|v.n| = {wall_normal:.2e}  ({normal_ratio:.2f} of interior speed, tol {slip_tolerance:.2f})")

    print(f"  overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def plot(sdf, velx, vely, divergence, output_path, show):
    extent = [DOMAIN_X_MIN, DOMAIN_X_MAX, DOMAIN_Y_MIN, DOMAIN_Y_MAX]
    imshow_kwargs = dict(extent=extent, origin="lower", aspect="equal")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    handle = axes[0].imshow(sdf, cmap="RdBu_r", **imshow_kwargs)
    axes[0].contour(sdf, levels=[0.0], colors="k", linewidths=1.0, extent=extent)
    axes[0].set_title("SDF (circle bubble): red = vapor")
    figure.colorbar(handle, ax=axes[0], fraction=0.046)

    speed = torch.sqrt(velx**2 + vely**2)
    handle = axes[1].imshow(speed, cmap="viridis", **imshow_kwargs)
    height, width = velx.shape
    step = max(height, width) // 24
    grid_y, grid_x = domain_grid(height, width)
    axes[1].quiver(
        grid_x[::step, ::step], grid_y[::step, ::step],
        velx[::step, ::step], vely[::step, ::step],
        color="white", scale_units="xy", angles="xy",
    )
    axes[1].contour(sdf, levels=[0.0], colors="k", linewidths=0.8, extent=extent)
    axes[1].set_title("velocity from potentials (magnitude + quiver)")
    figure.colorbar(handle, ax=axes[1], fraction=0.046)

    limit = divergence.abs().max().item() or 1.0
    handle = axes[2].imshow(divergence, cmap="RdBu_r", vmin=-limit, vmax=limit, **imshow_kwargs)
    axes[2].contour(sdf, levels=[0.0], colors="k", linewidths=0.8, extent=extent)
    axes[2].set_title("divergence: ~0 in liquid and near walls, nonzero at interface")
    figure.colorbar(handle, ax=axes[2], fraction=0.046)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"saved figure to {output_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--radius", type=float, default=3.0, help="bubble radius in domain units")
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=5.0)
    parser.add_argument("--band", type=float, default=2.0, help="vapor-gate ramp width in sdf units")
    parser.add_argument("--num-modes", type=int, default=6, help="Fourier modes per random potential")
    parser.add_argument("--physical-psi", action="store_true",
                        help="make the random streamfunction vanish at the walls (like a real "
                             "streamfunction); the near-wall velocity is then clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="max |div| (as a fraction of the interface peak) allowed at walls / in liquid")
    parser.add_argument("--slip-tolerance", type=float, default=0.4,
                        help="max wall-normal speed (as a fraction of interior speed) for free-slip")
    parser.add_argument("--output", type=Path, default=Path("divergence.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.seed)

    sdf = circle_sdf(args.height, args.width, args.radius, args.center_x, args.center_y)
    gate = band_mask(sdf, args.band).to(sdf.dtype)
    psi = smooth_random_field(args.height, args.width, args.num_modes, generator)
    phi = smooth_random_field(args.height, args.width, args.num_modes, generator)
    if args.physical_psi:
        psi = psi * wall_vanishing_envelope(args.height, args.width)

    velx, vely = velocity_from_potentials(psi, phi, gate)

    dx = (DOMAIN_X_MAX - DOMAIN_X_MIN) / args.width
    dy = (DOMAIN_Y_MAX - DOMAIN_Y_MIN) / args.height
    divergence = pointwise_divergence(velx, vely, dx, dy)

    all_pass = check_boundary_conditions(velx, vely, divergence, sdf, args.band, args.tolerance, args.slip_tolerance)
    plot(sdf, velx, vely, divergence, args.output, args.show)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
