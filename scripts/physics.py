"""Visualize the physics operators on a single simulation frame: the Helmholtz
decomposition of the (staggered) velocity and the interfacial mass-transfer field.

    python scripts/physics.py --path <sim.hdf5> --frame 100 --output <dir>

The HDF5 is a BubbleML_staggered file with face velocities ``velfacex`` (H, W+1)
and ``velfacey`` (H+1, W), plus cell-centered ``dfun`` (SDF) and ``temperature``.
The matching ``<sim>.json`` supplies the fluid parameters for the mass transfer.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from nucleus.physics.poisson import (
    helmholtz_from_faces,
    reconstruct_velocity_from_helmholtz,
    curl_faces_from_nodes,
    grad_faces_from_centers,
    divergence_centers_from_faces,
)
from nucleus.physics.mass_transfer import mass_transfer, continuity

# Fixed-resolution spacing (matches the rest of the codebase). TODO: derive from
# the sim params as (x_max - x_min) / (num_blocks_x * nx_block).
DX = DY = 1 / 32


def load_frame(path, frame):
    with h5py.File(path, "r") as handle:
        velfacex = torch.tensor(handle["velfacex"][frame], dtype=torch.float64)
        velfacey = torch.tensor(handle["velfacey"][frame], dtype=torch.float64)
        sdf = torch.tensor(handle["dfun"][frame], dtype=torch.float64)
        temp = torch.tensor(handle["temperature"][frame], dtype=torch.float64)
        mass_flux = torch.tensor(handle["massflux"][frame], dtype=torch.float64)
    return velfacex, velfacey, sdf, temp, mass_flux


def load_sim_params(path):
    with open(str(path).replace(".hdf5", ".json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def faces_to_centers(facex, facey):
    """Average the bounding faces of each cell so a face vector field can be shown
    as a cell-centered magnitude."""
    center_x = 0.5 * (facex[..., :, 1:] + facex[..., :, :-1])
    center_y = 0.5 * (facey[..., 1:, :] + facey[..., :-1, :])
    return center_x, center_y


def _magnitude(facex, facey):
    center_x, center_y = faces_to_centers(facex, facey)
    return torch.sqrt(center_x**2 + center_y**2).numpy()


def _draw_interface(ax, sdf):
    ax.contour(sdf, levels=[0.0], colors="k", linewidths=0.6)


def _symlog_norm(field, floor_decades=4):
    """Symmetric-log norm for a signed field: log scaling away from zero, linear
    within a small threshold through it. ``floor_decades`` sets how many decades
    below the peak magnitude are resolved before the linear region takes over."""
    limit = max(float(np.abs(field).max()), 1e-30)
    linear_threshold = limit * 10.0 ** (-floor_decades)
    return SymLogNorm(linthresh=linear_threshold, vmin=-limit, vmax=limit, base=10)


def plot_helmholtz(velfacex, velfacey, sdf, save_path):
    """Two rows: the magnitude of each field (velocity, its solenoidal and
    dilatational parts, the reconstruction) and the magnitude of each field's
    exact (MAC) divergence."""
    psi, phi = helmholtz_from_faces(velfacex, velfacey, DX, DY)
    curl_x, curl_y = curl_faces_from_nodes(psi, DX, DY)
    grad_x, grad_y = grad_faces_from_centers(phi, DX, DY)
    recon_x, recon_y = reconstruct_velocity_from_helmholtz(psi, phi, DX, DY)

    residual = _magnitude(velfacex - recon_x, velfacey - recon_y)
    speed = _magnitude(velfacex, velfacey)
    print(f"helmholtz relative reconstruction residual: {np.linalg.norm(residual) / np.linalg.norm(speed):.3e}")

    columns = [
        ("velocity", velfacex, velfacey),
        ("solenoidal curl(psi)", curl_x, curl_y),
        ("dilatational grad(phi)", grad_x, grad_y),
        ("reconstruction", recon_x, recon_y),
    ]
    sdf_np = sdf.numpy()

    fig, axes = plt.subplots(2, len(columns), figsize=(4 * len(columns), 8), sharex=True, sharey=True)
    for col, (title, facex, facey) in enumerate(columns):
        top = axes[0, col].imshow(_magnitude(facex, facey), origin="lower", cmap="magma")
        axes[0, col].set_title(title)
        _draw_interface(axes[0, col], sdf_np)
        fig.colorbar(top, ax=axes[0, col], fraction=0.046)

        # Divergence spans orders of magnitude -> log-scale with a per-panel floor.
        div = np.abs(divergence_centers_from_faces(facex, facey, DX, DY).numpy())
        vmax = max(float(div.max()), 1e-30)
        bottom = axes[1, col].imshow(
            div, origin="lower", cmap="viridis",
            norm=LogNorm(vmin=vmax * 1e-6, vmax=vmax, clip=True),
        )
        _draw_interface(axes[1, col], sdf_np)
        fig.colorbar(bottom, ax=axes[1, col], fraction=0.046)

    axes[0, 0].set_ylabel("magnitude")
    axes[1, 0].set_ylabel("|divergence|")
    fig.suptitle("Helmholtz decomposition:  u = curl(psi) + grad(phi)")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


def non_dimensionalize_temp(temp, bulk_temp, heater_temp):
    """Map temperature to the non-dimensional (bulk -> 0, heater -> 1) scale the
    Re/Pr/Stefan mass-transfer expression is written in."""
    return (temp - bulk_temp) / (heater_temp - bulk_temp)


def _zero_mean_correlation(first, second):
    """Pearson correlation of two flattened fields; 0 when either is constant."""
    first = first.ravel() - first.mean()
    second = second.ravel() - second.mean()
    denom = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denom) if denom > 0 else 0.0


def _band_profile(computed_mdot, data_massflux, sdf, spacing):
    """Print the mean flux vs. distance from the interface, in one-cell bins of
    ``|sdf|``. This separates the interface match (bin 0) from the band profile
    (outer bins): if bin 0 agrees but the outer bins diverge, the interface is
    right and only the spreading across the band is off."""
    distance_in_cells = (np.abs(sdf) / spacing).astype(int)
    print("  dist  |   mean mdot    mean massflux    ratio")
    for band in range(6):
        cells = distance_in_cells == band
        if not np.any(cells):
            continue
        mdot_mean = computed_mdot[cells].mean()
        data_mean = data_massflux[cells].mean()
        ratio = mdot_mean / data_mean if abs(data_mean) > 1e-30 else float("nan")
        print(f"  {band:>2d}dx  | {mdot_mean:+.3e}   {data_mean:+.3e}    {ratio:+.3f}")


def _near_heater_profile(computed_mdot, data_massflux, strip=20):
    """Report the flux ratio and correlation in horizontal strips by distance from
    the bottom wall (the heater at row 0), to localize the near-heater error: a
    large ratio / low correlation in the first strips means the calculation is off
    specifically near the heater."""
    height, width = computed_mdot.shape[-2:]
    rows = np.broadcast_to(np.arange(height)[:, None], (height, width))
    mask_data = np.abs(data_massflux) > 1e-6 * np.abs(data_massflux).max()
    print("  rows (from heater) | ratio (sum/sum)   correlation   ncells")
    for start in range(0, height, strip):
        strip_cells = mask_data & (rows >= start) & (rows < start + strip)
        count = int(strip_cells.sum())
        if count == 0:
            continue
        computed = computed_mdot[strip_cells]
        data = data_massflux[strip_cells]
        ratio = computed.sum() / (data.sum() + 1e-30)
        correlation = _zero_mean_correlation(computed, data)
        print(f"  {start:>3d}-{start + strip:<4d}           | {ratio:+8.3f}         {correlation:+.3f}        {count}")


def diagnose_mass_transfer(computed_mdot, data_massflux, sdf=None, spacing=None):
    """Print diagnostics that localize *why* the computed Stefan mdot differs from
    the Flash-X massflux, separating the failure modes so they point to different
    fixes:

      * scale     -- a single best-fit factor (units / missing non-dimensional
                     group / density factor). ``alpha`` far from +-1.
      * sign      -- ``alpha`` negative, or low sign agreement on the overlap.
      * support   -- the two are nonzero in different cells (Jaccard << 1); the
                     interface band is placed or widened differently.
      * grid shift -- correlation jumps when ``mdot`` is rolled by a cell, i.e. a
                     half/one-cell offset between the two discretizations.

    Both inputs are cell-centered numpy arrays of shape ``(H, W)``.
    """
    print("\n=== mass-transfer diagnostics ===")
    for name, field in (("computed mdot", computed_mdot), ("data massflux", data_massflux)):
        print(
            f"{name:14s}: min={field.min():+.3e}  max={field.max():+.3e}  "
            f"mean={field.mean():+.3e}  L2={np.linalg.norm(field):.3e}  "
            f"nnz={np.count_nonzero(field)}"
        )

    # Integrated transfer: matches if only the spatial spreading differs.
    net_ratio = computed_mdot.sum() / (data_massflux.sum() + 1e-30)
    print(f"net ratio (sum mdot / sum massflux): {net_ratio:+.3e}")

    # Support overlap: are the two nonzero in the same cells? A low Jaccard with a
    # large data-only count means the data spreads the source over a wider band.
    mask_mdot = computed_mdot != 0
    mask_data = np.abs(data_massflux) > 1e-6 * np.abs(data_massflux).max()
    intersection = np.count_nonzero(mask_mdot & mask_data)
    union = np.count_nonzero(mask_mdot | mask_data)
    print(
        f"support Jaccard: {intersection / max(union, 1):.3f}  "
        f"(mdot-only={np.count_nonzero(mask_mdot & ~mask_data)}, "
        f"data-only={np.count_nonzero(~mask_mdot & mask_data)})"
    )

    # Best-fit scale and correlation on the data's support: high correlation with
    # alpha != 1 is a pure scale/sign problem; low correlation is a shape problem.
    if np.any(mask_data):
        computed = computed_mdot[mask_data]
        data = data_massflux[mask_data]
        alpha = float(computed @ data / (computed @ computed + 1e-30))
        agree = np.count_nonzero(np.sign(computed) == np.sign(data))
        print(f"best-fit scale (massflux ~ alpha * mdot): alpha={alpha:+.3e}")
        print(f"correlation on data support: {_zero_mean_correlation(computed, data):+.3f}")
        print(f"sign agreement on data support: {agree}/{computed.size}")

    # Integer-shift search: if rolling mdot by a cell sharply improves correlation,
    # the two live on offset grids (e.g. face- vs cell-centered density gradient).
    no_shift = _zero_mean_correlation(computed_mdot, data_massflux)
    best_shift, best_corr = (0, 0), no_shift
    for shift_y in (-1, 0, 1):
        for shift_x in (-1, 0, 1):
            rolled = np.roll(np.roll(computed_mdot, shift_y, axis=-2), shift_x, axis=-1)
            corr = _zero_mean_correlation(rolled, data_massflux)
            if corr > best_corr:
                best_shift, best_corr = (shift_y, shift_x), corr
    print(f"correlation: {no_shift:+.3f} (no shift) -> {best_corr:+.3f} at (dy,dx)={best_shift}")
    
    print("Absmax Error: ", abs(computed_mdot - data_massflux).max())
    print("Absmean band Error: ", abs(computed_mdot - data_massflux)[abs(data_massflux) > 0].mean())

    # Per-distance profile: shows whether the interface (bin 0) matches while the
    # band (outer bins) diverges.
    if sdf is not None and spacing is not None:
        _band_profile(computed_mdot, data_massflux, sdf, spacing)
    # Distance-from-heater profile: shows whether the error localizes near the wall.
    _near_heater_profile(computed_mdot, data_massflux)
    print("=================================\n")


def plot_mass_transfer(temp, sdf, mass_flux, velfacex, velfacey, sim_params, save_path, band_cells, use_wall_bc=True):
    """The computed interfacial mass-transfer field, the velocity divergence it
    should match, the Flash-X massflux, and the difference to Flash-X."""
    # The mass-transfer expression is non-dimensional, so the temperature and the
    # saturation temperature must be on the same (bulk -> 0, heater -> 1) scale.
    bulk_temp = sim_params["bulk_temp"]
    heater_temp = sim_params["heater"]["wallTemp"]
    temp = non_dimensionalize_temp(temp.to(torch.float64), bulk_temp, heater_temp)
    sat_temp = non_dimensionalize_temp(sim_params["sat_temp"], bulk_temp, heater_temp)
    # The heater wall is at heater_temp, which is 1 on the non-dimensional scale.
    wall_temp = non_dimensionalize_temp(heater_temp, bulk_temp, heater_temp) if use_wall_bc else None

    mdot = mass_transfer(
        temp.to(torch.float64), sdf.to(torch.float64),
        sat_temp=float(sat_temp),
        dx=DX, dy=DY,
        stefan=sim_params["stefan"],
        reynolds=1.0 / sim_params["inv_reynolds"],
        prandtl=sim_params["prandtl"],
        thermal_conductivity=sim_params["thcogas"],
        band_cells=band_cells,
        wall_temp=wall_temp,
    ).numpy()
    total = mdot.sum() * DX * DY
    mass_flux = mass_flux.numpy()
    difference = mdot - mass_flux
    # Velocity divergence (exact MAC divergence from the face velocities): the
    # phase-change source should show up here, so mass transfer ~ div(u).
    divergence = divergence_centers_from_faces(
        velfacex.to(torch.float64), velfacey.to(torch.float64), DX, DY
    ).numpy()
    print(f"net mass transfer (sum * dx * dy): {total:.4e}   (negative => net evaporation)")
    div_ratio = mdot.sum() / (divergence.sum() + 1e-30)
    print(f"mass transfer vs velocity divergence: net ratio={div_ratio:+.3e}  "
          f"correlation={_zero_mean_correlation(mdot, divergence):+.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)

    mdot_img = axes[0].imshow(mdot, origin="lower", cmap="RdBu_r", norm=_symlog_norm(mdot))
    axes[0].set_title("mass transfer (blue: evap, red: cond)")
    fig.colorbar(mdot_img, ax=axes[0], fraction=0.046)

    divergence_img = axes[1].imshow(divergence, origin="lower", cmap="RdBu_r", norm=_symlog_norm(divergence))
    axes[1].set_title("velocity divergence  div(u)")
    fig.colorbar(divergence_img, ax=axes[1], fraction=0.046)

    mass_flux_img = axes[2].imshow(mass_flux, origin="lower", cmap="RdBu_r", norm=_symlog_norm(mass_flux))
    axes[2].set_title("Flash-X mass transfer (blue: evap, red: cond)")
    fig.colorbar(mass_flux_img, ax=axes[2], fraction=0.046)

    difference_img = axes[3].imshow(difference, origin="lower", cmap="RdBu", norm=_symlog_norm(difference))
    axes[3].set_title("computed − Flash-X")
    fig.colorbar(difference_img, ax=axes[3], fraction=0.046)

    diagnose_mass_transfer(mdot, mass_flux, sdf=sdf.numpy(), spacing=DX)

    fig.suptitle("Interfacial mass transfer")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")


def plot_continuity_vs_divergence(velfacex, velfacey, temp, sdf, sim_params, save_path, band_cells, use_wall_bc=True):
    """Compare the physical continuity field (mdot * grad(rho).n) against the
    divergence of the reconstructed velocity.  Both fields live on the
    cell-centered grid."""
    # -- continuity (physical source from the Stefan condition) --
    bulk_temp = sim_params["bulk_temp"]
    heater_temp = sim_params["heater"]["wallTemp"]
    temp_nd = non_dimensionalize_temp(temp, bulk_temp, heater_temp)
    sat_temp = non_dimensionalize_temp(sim_params["sat_temp"], bulk_temp, heater_temp)
    wall_temp = non_dimensionalize_temp(heater_temp, bulk_temp, heater_temp) if use_wall_bc else None

    cont = continuity(
        temp_nd, sdf,
        sat_temp=sat_temp,
        dx=DX, dy=DY,
        stefan=sim_params["stefan"],
        reynolds=1.0 / sim_params["inv_reynolds"],
        prandtl=sim_params["prandtl"],
        thermal_conductivity=sim_params["thcogas"],
        rhogas=sim_params["rhogas"],
        band_cells=band_cells,
        wall_temp=wall_temp,
    ).numpy()

    divergence = divergence_centers_from_faces(velfacex, velfacey, 1/32, 1/32).numpy()

    sdf_np = sdf.numpy()
    abs_diff = np.abs(cont - divergence)

    rel_error = np.linalg.norm(abs_diff) / np.linalg.norm(cont)
    print(f"continuity vs. velocity divergence relative L2 error: {rel_error:.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    im0 = axes[0].imshow(cont, origin="lower", cmap="RdBu_r", norm=_symlog_norm(cont))
    axes[0].set_title("continuity")
    #_draw_interface(axes[0], sdf_np)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(divergence, origin="lower", cmap="RdBu_r", norm=_symlog_norm(divergence))
    axes[1].set_title("velocity divergence")
    #_draw_interface(axes[1], sdf_np)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    # abs_diff is non-negative -> plain log scale with a per-panel floor.
    diff_max = max(float(abs_diff.max()), 1e-30)
    im2 = axes[2].imshow(
        abs_diff, origin="lower", cmap="viridis",
        norm=LogNorm(vmin=diff_max * 1e-6, vmax=diff_max, clip=True),
    )
    axes[2].set_title("|continuity − divergence|")
    #_draw_interface(axes[2], sdf_np)
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle("Continuity vs. velocity divergence")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {save_path}")
    
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, type=str, help="BubbleML_staggered .hdf5 file")
    parser.add_argument("--frame", type=int, default=100, help="time index to visualize")
    parser.add_argument("--output", type=str, default=".", help="directory to write figures to")
    parser.add_argument(
        "--band-cells", type=int, default=3,
        help="half-width (cells) the Stefan flux is extrapolated/spread over; "
             "increase to match the width of the Flash-X massflux band",
    )
    parser.add_argument(
        "--no-wall-bc", action="store_true",
        help="disable the Dirichlet heater BC on the ghost-fluid gradients "
             "(use a zero-gradient wall instead), to A/B test the contact-line error",
    )
    args = parser.parse_args()

    path = Path(args.path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    velfacex, velfacey, sdf, temp, mass_flux = load_frame(path, args.frame)
    sim_params = load_sim_params(path)
    use_wall_bc = not args.no_wall_bc

    plot_helmholtz(velfacex, velfacey, sdf, output / "helmholtz.png")
    plot_mass_transfer(temp, sdf, mass_flux, velfacex, velfacey, sim_params, output / "mass_transfer.png", args.band_cells, use_wall_bc)
    plot_continuity_vs_divergence(
        velfacex, velfacey, temp, sdf, sim_params, output / "continuity_vs_divergence.png",
        args.band_cells, use_wall_bc,
    )


if __name__ == "__main__":
    main()
