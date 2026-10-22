"""Half-disk and half-annulus illumination with a circular NA=0.75 detector.

Run: uv run python examples/half_patterns.py
Lengths are in um, wavevectors in rad/um, arrays in (z,y,x) FFT order.
The upper half-plane is ky >= 0; the annulus includes NA=0.5 and NA=0.75.
Both the selected method (FFT by default) and linear Ewald are plotted.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from wotf import circular_pupil, compute_wotf, k_axes, precision


def run_example(dtype="float64", method="fft", axial="finite", output=Path("artifacts/half-patterns")):
    """Return (selected-method WOTFs, linear Ewald WOTFs), keyed by pattern."""
    shape = (129, 129, 129)
    spacing = (0.35, 0.15, 0.15)  # sample-space (dz,dy,dx), um
    wavelength, n_medium = 0.53, 1.33
    na_outer, na_inner = 0.75, 0.5

    with precision(dtype):
        kz, ky, kx = k_axes(shape, spacing)
        pupil = circular_pupil(ky, kx, wavelength=wavelength, na=na_outer)
        radial2 = ky[:, None] ** 2 + kx[None, :] ** 2
        half_disk = (pupil != 0) & (ky[:, None] >= 0)
        half_annulus = half_disk & (radial2 >= (2 * jnp.pi * na_inner / wavelength) ** 2)
        sources = {
            "Half disk, NA 0-0.75": half_disk.astype(ky.dtype),
            "Half annulus, NA 0.5-0.75": half_annulus.astype(ky.dtype),
        }
        transfers, linear_transfers = {}, {}
        for name, source in sources.items():
            transfers[name] = compute_wotf(
                shape, spacing, wavelength=wavelength, n_medium=n_medium,
                source=source, pupil=pupil, method=method, axial=axial,
            )
            if method == "ewald" and axial == "linear":
                linear_transfers[name] = transfers[name]
            else:
                linear_transfers[name] = compute_wotf(
                    shape, spacing, wavelength=wavelength, n_medium=n_medium,
                    source=source, pupil=pupil, method="ewald", axial="linear",
                )
            h_real, h_imag = transfers[name]
            background = jnp.sum(source * jnp.abs(pupil) ** 2)
            # For a real pupil and an odd, symmetric axial lag grid, H_real
            # is imaginary and H_imag is real. Report the unused components.
            residual = max(float(jnp.max(jnp.abs(h_real.real))), float(jnp.max(jnp.abs(h_imag.imag))))
            print(f"{name}: source modes={int(jnp.count_nonzero(source))}, I0={float(background):g}, "
                  f"dtype={h_real.dtype}, other-component max={residual:.3e} um^2")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected_label = "FFT" if method == "fft" else f"Ewald ({axial})"
    _plot(output, (kz, ky, kx), sources, pupil, transfers, linear_transfers,
          wavelength, n_medium, na_outer, selected_label)
    for filename in ("patterns.png", "transfer_functions.png", "transfer_functions_linear.png"):
        print(f"Saved {output / filename}")
    return transfers, linear_transfers


def _plot(output, axes_k, sources, pupil, transfers, linear_transfers, wavelength, n_medium, na_outer, selected_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    kz, ky, kx = [np.fft.fftshift(np.asarray(axis)) for axis in axes_k]
    k0 = 2 * np.pi / wavelength
    dkz = kz[1] - kz[0]
    # Crop the displayed axial range to the cap differences plus two bins;
    # the actual computed WOTFs retain their entire FFT frequency grid.
    kz_limit = k0 * n_medium - np.sqrt((k0 * n_medium) ** 2 - (k0 * na_outer) ** 2) + 2 * dkz

    fig, panels = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")
    na_extent = ((kx[0] - (kx[1] - kx[0]) / 2) / k0, (kx[-1] + (kx[1] - kx[0]) / 2) / k0,
                 (ky[0] - (ky[1] - ky[0]) / 2) / k0, (ky[-1] + (ky[1] - ky[0]) / 2) / k0)
    patterns = list(sources.values()) + [np.abs(np.asarray(pupil))]
    pattern_labels = (r"$S_{\mathrm{disk}}$", r"$S_{\mathrm{annulus}}$", r"$|P|$")
    for panel, pattern, label in zip(panels, patterns, pattern_labels):
        panel.imshow(np.fft.fftshift(np.asarray(pattern)), origin="lower", extent=na_extent,
                     aspect="equal", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        panel.set(title=label, xlabel=r"$k_x/k_0$", xlim=(-0.85, 0.85), ylim=(-0.85, 0.85))
    panels[0].set_ylabel(r"$k_y/k_0$")
    fig.savefig(output / "patterns.png", dpi=160)
    plt.close(fig)

    # Shift only for display. H_real acts on F_real, H_imag acts on F_imag;
    # their names do not mean the real/imaginary parts of one transfer array.
    displayed = [
        {name: (np.fft.fftshift(np.asarray(h_real)).imag, np.fft.fftshift(np.asarray(h_imag)).real)
         for name, (h_real, h_imag) in result.items()}
        for result in (transfers, linear_transfers)
    ]
    limits = [max(np.max(np.abs(pair[channel])) for result in displayed for pair in result.values())
              for channel in (0, 1)]
    # Show weak cap structure together with the large absorption DC response;
    # preserve signs and actual values, with a linear scale near zero.
    norms = [SymLogNorm(linthresh=1e-3 * limit, vmin=-limit, vmax=limit) for limit in limits]
    for data, filename, title in zip(
        displayed, ("transfer_functions.png", "transfer_functions_linear.png"),
        (selected_label, "Ewald (linear)"),
    ):
        _plot_transfers(output / filename, (kz, ky, kx), data, norms,
                        kz_limit, 2 * k0 * na_outer, title)


def _plot_transfers(path, axes_k, displayed, norms, kz_limit, lateral_limit, title):
    import matplotlib.pyplot as plt

    kz, ky, kx = axes_k
    nz, ny, nx = next(iter(displayed.values()))[0].shape
    coordinates = ((kx, ky), (ky, kz), (kx, kz))
    unit = r"\;[\mathrm{rad}\,\mu\mathrm{m}^{-1}]"
    labels = ((rf"$k_x{unit}$", rf"$k_y{unit}$"),
              (rf"$k_y{unit}$", rf"$k_z{unit}$"),
              (rf"$k_x{unit}$", rf"$k_z{unit}$"))
    planes = (r"$k_z=0$", r"$k_x=0$", r"$k_y=0$")
    channels = (r"$\operatorname{Im}H_{\mathrm{real}}$", r"$\operatorname{Re}H_{\mathrm{imag}}$")
    # Match row heights to the physical frequency ranges. Equal aspect
    # gives one rad/um the same display length along kz, ky, and kx.
    axial_ratio = kz_limit / lateral_limit
    fig = plt.figure(figsize=(13, 7), layout="constrained")
    fig.suptitle(title, fontsize=12)
    grid = fig.add_gridspec(5, 4, height_ratios=(0.12, 1, axial_ratio, axial_ratio, 0.045))
    for channel in (0, 1):
        heading = fig.add_subplot(grid[0, 2 * channel:2 * channel + 2])
        heading.set_axis_off()
        heading.text(0.5, 0.5, channels[channel], ha="center", va="center", fontsize=13,
                     transform=heading.transAxes)
        for pattern_index, (pair, name) in enumerate(zip(displayed.values(), ("Half disk", "Half annulus"))):
            column = 2 * channel + pattern_index
            data = pair[channel]
            slices = (data[nz // 2], data[:, :, nx // 2], data[:, ny // 2, :])
            for row, (image, (horizontal, vertical), (xlabel, ylabel), plane) in enumerate(
                zip(slices, coordinates, labels, planes)
            ):
                panel = fig.add_subplot(grid[row + 1, column])
                dh, dv = horizontal[1] - horizontal[0], vertical[1] - vertical[0]
                extent = (horizontal[0] - dh / 2, horizontal[-1] + dh / 2,
                          vertical[0] - dv / 2, vertical[-1] + dv / 2)
                im = panel.imshow(image, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r",
                                  norm=norms[channel], interpolation="nearest")
                panel.set_xlabel(xlabel)
                panel.set_xlim(-lateral_limit, lateral_limit)
                panel.set_ylim((-lateral_limit, lateral_limit) if row == 0 else (-kz_limit, kz_limit))
                if row == 0:
                    panel.set_title(name, fontsize=10)
                if column == 0:
                    panel.set_ylabel(ylabel + "\n" + plane)
                else:
                    panel.tick_params(labelleft=False)
        colorbar = fig.add_subplot(grid[4, 2 * channel:2 * channel + 2])
        bar = fig.colorbar(im, cax=colorbar, orientation="horizontal", label=r"$\mu\mathrm{m}^{2}$ (symlog)")
        # Keep the central zero without crowding it with small linear ticks.
        bar.set_ticks([tick for tick in bar.get_ticks()
                       if tick == 0 or norms[channel].linthresh <= abs(tick) <= norms[channel].vmax])
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--method", choices=("fft", "ewald"), default="fft")
    parser.add_argument("--axial", choices=("finite", "linear"), default="finite")
    parser.add_argument("--output", type=Path, default=Path("artifacts/half-patterns"))
    args = parser.parse_args()
    run_example(args.dtype, args.method, args.axial, args.output)
