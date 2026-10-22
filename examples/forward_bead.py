"""WOTF forward prediction of a weak, slightly absorbing spherical bead.

Run: uv run python examples/forward_bead.py
All lengths are in um. This is a scalar WOTF demonstration, not a Mie model.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from wotf import circular_pupil, compute_wotf, k_axes, precision, predict_contrast


def make_bead(shape=(65, 65, 65), *, dtype="float64"):
    """Centered radius-0.75 um sphere; return F_real, F_imag, spacing, optics."""
    spacing = (0.25, 0.15, 0.15)
    wavelength, n_medium = 0.53, 1.33
    n_bead, radius = 1.331 + 2e-5j, 0.75
    with precision(dtype):
        _, ky, kx = k_axes(shape, spacing)
        z, y, x = [(jnp.arange(n, dtype=ky.dtype) - n // 2) * d for n, d in zip(shape, spacing)]
        inside = z[:, None, None] ** 2 + y[None, :, None] ** 2 + x[None, None, :] ** 2 <= radius**2
        potential = inside * ((2 * jnp.pi / wavelength) ** 2 * (n_medium**2 - n_bead**2))
        pupil = circular_pupil(ky, kx, wavelength=wavelength, na=0.75)
        source = ((pupil != 0) & (ky[:, None] >= 0)).astype(ky.dtype)
        optics = dict(wavelength=wavelength, n_medium=n_medium, source=source, pupil=pupil)
        return potential.real, potential.imag, spacing, optics


def simulate_bead(shape=(65, 65, 65), *, dtype="float64", method="fft", axial="finite"):
    """Predict a centered contrast/intensity stack using a half-disk source."""
    with precision(dtype):
        fr, fi, spacing, optics = make_bead(shape, dtype=dtype)
        hr, hi = compute_wotf(shape, spacing, **optics, method=method, axial=axial)
        contrast = predict_contrast(fr, fi, hr, hi)
        background = jnp.sum(optics["source"] * jnp.abs(optics["pupil"]) ** 2)
        return dict(
            F_real=fr, F_imag=fi, H_real=hr, H_imag=hi,
            contrast=contrast, intensity=background * (1 + contrast),
            background=background, spacing=spacing,
        )


def run_example(dtype="float64", method="fft", axial="finite", output=Path("artifacts/bead")):
    result = simulate_bead(dtype=dtype, method=method, axial=axial)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "bead_stack.npz", **{
        key: np.asarray(result[key])
        for key in ("F_real", "F_imag", "contrast", "intensity", "background", "spacing")
    })
    label = "FFT" if method == "fft" else f"Ewald ({axial})"
    _plot(output / "bead_forward.png", result, label)
    c = np.asarray(result["contrast"])
    print(f"{label}, {c.dtype}: I0={float(result['background']):g}, contrast=[{c.min():.6g}, {c.max():.6g}]")
    print(f"Saved {output / 'bead_forward.png'} and {output / 'bead_stack.npz'}")
    return result


def _plot(path, result, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr, contrast = np.asarray(result["F_real"]), np.asarray(result["contrast"])
    nz, ny, nx = fr.shape
    z, y, x = [(np.arange(n) - n // 2) * d for n, d in zip(fr.shape, result["spacing"])]
    coordinates = ((x, y), (y, z), (x, z))
    labels = ((r"$x\;[\mu\mathrm{m}]$", r"$y\;[\mu\mathrm{m}]$"),
              (r"$y\;[\mu\mathrm{m}]$", r"$z\;[\mu\mathrm{m}]$"),
              (r"$x\;[\mu\mathrm{m}]$", r"$z\;[\mu\mathrm{m}]$"))
    planes = (r"$z=0$", r"$x=0$", r"$y=0$")
    limits = ((fr.min(), 0), (-np.max(np.abs(contrast)), np.max(np.abs(contrast))))
    color_labels = (r"$F_{\mathrm{real}}\;[\mu\mathrm{m}^{-2}]$", r"$c=(I-I_0)/I_0$")
    fig, panels = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
    fig.suptitle(f"Bead forward: {title}")
    for row, (volume, cmap) in enumerate(((fr, "gray"), (contrast, "RdBu_r"))):
        slices = (volume[nz // 2], volume[:, :, nx // 2], volume[:, ny // 2, :])
        for column, (image, (horizontal, vertical), (xlabel, ylabel)) in enumerate(zip(slices, coordinates, labels)):
            dh, dv = horizontal[1] - horizontal[0], vertical[1] - vertical[0]
            extent = (horizontal[0] - dh / 2, horizontal[-1] + dh / 2,
                      vertical[0] - dv / 2, vertical[-1] + dv / 2)
            panel = panels[row, column]
            im = panel.imshow(image, origin="lower", extent=extent, aspect="equal", cmap=cmap,
                              vmin=limits[row][0], vmax=limits[row][1], interpolation="nearest")
            # Crop only the display: |focus z - specimen z| <= 3 um here,
            # within the padded lag range of the default 65-plane volume.
            panel.set(xlabel=xlabel, ylabel=ylabel, xlim=(-2.25, 2.25), ylim=(-2.25, 2.25))
            if row == 0:
                panel.set_title(planes[column])
        fig.colorbar(im, ax=panels[row, :], label=color_labels[row])
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--method", choices=("fft", "ewald"), default="fft")
    parser.add_argument("--axial", choices=("finite", "linear"), default="finite")
    parser.add_argument("--output", type=Path, default=Path("artifacts/bead"))
    args = parser.parse_args()
    run_example(args.dtype, args.method, args.axial, args.output)
