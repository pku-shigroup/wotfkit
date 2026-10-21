"""Independent slice-integrated Born fields versus WOTF predictions.

Run with ``uv run python examples/validate_born.py --dtype float64``.
The reference below imports no wotf functions, including grid/precision helpers.
It forms fields at an exit plane before propagating them to focus positions;
it never forms a 3D object spectrum, a correlation, or a transfer function.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp


def make_case(shape=(17, 15, 17), *, dtype="float64", aberrated=True):
    """Known F in um^-2, five occupied axial layers; all lengths in um."""
    if dtype not in ("float32", "float64"):
        raise ValueError("unsupported dtype")
    real = getattr(jnp, dtype)
    with jax.enable_x64(True):
        spacing = (0.35, 0.25, 0.22)
        nz, ny, nx = shape
        z, y, x = [
            (jnp.arange(n, dtype=real) - n // 2) * d
            for n, d in zip(shape, spacing)
        ]
        Z, Y, X = jnp.meshgrid(z, y, x, indexing="ij")
        fr = -0.01 * jnp.exp(-((X - 0.18) ** 2 / 0.5 + (Y + 0.12) ** 2 / 0.4 + Z**2 / 0.13))
        fi = -0.004 * jnp.exp(-((X + 0.2) ** 2 / 0.35 + (Y - 0.13) ** 2 / 0.7 + Z**2 / 0.2))
        occupied = jnp.abs(jnp.arange(nz) - nz // 2) <= 2
        fr, fi = fr * occupied[:, None, None], fi * occupied[:, None, None]
        fy = jnp.fft.fftfreq(ny, spacing[1], dtype=real)
        fx = jnp.fft.fftfreq(nx, spacing[2], dtype=real)
        FY, FX = jnp.meshgrid(fy, fx, indexing="ij")
        wavelength, n_medium, na = 0.53, 1.33, 0.6
        cutoff = na / wavelength
        phase = 0.25 * (FX / cutoff) ** 2 + 0.13 * FY / cutoff if aberrated else jnp.zeros_like(FX)
        pupil = (FX**2 + FY**2 <= cutoff**2) * jnp.exp(1j * phase)
        source = jnp.zeros((ny, nx), real).at[0, 0].set(1).at[1, 2].set(0.6).at[-1, 1].set(0.3)
        optics = dict(wavelength=wavelength, n_medium=n_medium, source=source, pupil=pupil)
        return fr, fi, spacing, optics


def born_stack(
    F_real, F_imag, spacing, *, wavelength, n_medium, source, pupil,
    z_focus, dtype="float64",
):
    """Independently integrate outgoing Born fields and return reference data.

    Centered spatial F has shape (nz,ny,nx); focus positions may be arbitrary.
    Transverse quadrature is periodic; the z integral is a finite slice sum.
    The medium Green function is exp(i km r)/(4 pi r), and F=k0^2(nm^2-n^2),
    hence the scattered field is MINUS G*(F*u_inc). Its forward angular
    spectrum factor is 1/(4 pi i w), w=sqrt((nm/lambda0)^2-fx^2-fy^2).

    Returned linear/Born intensities are absolute weighted intensities.
    Contrasts are also accumulated directly, avoiding subtraction of a large
    background when assessing weak signals. 'Born' includes |us|^2 but no
    multiple scattering. All numerical work uses the requested JAX precision.
    """
    if dtype not in ("float32", "float64"):
        raise ValueError("unsupported dtype")
    real = getattr(jnp, dtype)
    complex_ = jnp.complex128 if dtype == "float64" else jnp.complex64
    with jax.enable_x64(True):
        fr, fi = jnp.asarray(F_real, real), jnp.asarray(F_imag, real)
        nz, ny, nx = fr.shape
        dz, dy, dx = (jnp.asarray(d, real) for d in spacing)
        potential = fr + 1j * fi
        source, pupil = jnp.asarray(source, real), jnp.asarray(pupil, complex_)
        z = (jnp.arange(nz, dtype=real) - nz // 2) * dz
        y = (jnp.arange(ny, dtype=real) - ny // 2) * dy
        x = (jnp.arange(nx, dtype=real) - nx // 2) * dx
        Y, X = jnp.meshgrid(y, x, indexing="ij")
        fy, fx = jnp.fft.fftfreq(ny, dy, dtype=real), jnp.fft.fftfreq(nx, dx, dtype=real)
        radial2 = fy[:, None] ** 2 + fx[None, :] ** 2
        fm = jnp.asarray(n_medium / wavelength, real)
        active = (source > 0) | (jnp.abs(pupil) > 0)
        if bool(jnp.any(active & (radial2 >= fm**2))):
            raise ValueError("reference requires propagating active modes")
        w = jnp.sqrt(jnp.maximum(fm**2 - radial2, 0))
        safe_w = jnp.where(w > 0, w, 1)
        z_focus = jnp.asarray(z_focus, real)
        z_exit = z[-1] + dz  # strictly beyond every specimen slice
        sy, sx = jnp.nonzero(source > 0)
        background = jnp.sum(source * jnp.abs(pupil) ** 2)
        if not bool(background > 0):
            raise ValueError("nonzero reference background required")

        @jax.jit
        def integrate():
            def illuminate(i, accumulated):
                cross, power = accumulated
                ws = w[sy[i], sx[i]]
                lateral = jnp.exp(2j * jnp.pi * (fy[sy[i]] * Y + fx[sx[i]] * X))

                def scatter_slice(exit_spectrum, layer):
                    z_sample, f_slice = layer
                    driving = f_slice * lateral * jnp.exp(2j * jnp.pi * ws * z_sample)
                    # Continuous transverse transform -> dx dy FFT; inverse
                    # contributes 1/(dx dy), not a second voxel-volume factor.
                    driving_spectrum = dx * dy * jnp.fft.fft2(jnp.fft.ifftshift(driving))
                    contribution = (
                        driving_spectrum * dz / (4j * jnp.pi * safe_w)
                        * jnp.exp(2j * jnp.pi * w * (z_exit - z_sample))
                    )
                    return exit_spectrum + contribution, None

                exit_spectrum, _ = jax.lax.scan(
                    scatter_slice, jnp.zeros((ny, nx), complex_), (z, potential)
                )
                spectrum = (
                    pupil * exit_spectrum
                    * jnp.exp(2j * jnp.pi * w * (z_focus[:, None, None] - z_exit))
                )
                scattered = jnp.fft.fftshift(
                    jnp.fft.ifft2(spectrum), axes=(-2, -1)
                ) / (dx * dy)
                reference = (
                    pupil[sy[i], sx[i]] * lateral
                    * jnp.exp(2j * jnp.pi * ws * z_focus[:, None, None])
                )
                weight = source[sy[i], sx[i]]
                cross = cross + weight * 2 * jnp.real(jnp.conj(reference) * scattered)
                power = power + weight * jnp.abs(scattered) ** 2
                return cross, power

            zeros = jnp.zeros((z_focus.size, ny, nx), real)
            return jax.lax.fori_loop(0, sy.size, illuminate, (zeros, zeros))

        cross, power = integrate()
        return dict(
            linear_intensity=background + cross,
            born_intensity=background + cross + power,
            linear_contrast=cross / background,
            born_contrast=(cross + power) / background,
            background=background,
        )


def error_metrics(prediction, reference):
    """Contrast errors; the denominator is signal norm, never total intensity."""
    with jax.enable_x64(True):
        difference = jnp.asarray(prediction, jnp.float64) - reference
        denominator = jnp.linalg.norm(reference)
        return dict(
            relative_l2=float(jnp.linalg.norm(difference) / denominator),
            rmse=float(jnp.sqrt(jnp.mean(difference**2))),
            max_abs=float(jnp.max(jnp.abs(difference))),
        )


def run_validation(dtype="float64", output=Path("artifacts/born-float64")):
    # Library functions appear only on the prediction side of this comparison.
    from wotf import predict_contrast, wotf_ewald, wotf_fft

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with jax.enable_x64(True):
        fr, fi, spacing, optics = make_case()
        z_focus = jnp.arange(-4, 5, dtype=jnp.float64) * spacing[0]
        reference = born_stack(fr, fi, spacing, **optics, z_focus=z_focus)
        roi = slice(fr.shape[0] // 2 - 4, fr.shape[0] // 2 + 5)
        methods = {
            "fft": (wotf_fft, {}),
            "ewald_finite": (wotf_ewald, dict(axial="finite")),
            "ewald_linear": (wotf_ewald, dict(axial="linear")),
        }
        predictions, metrics = {}, {}
        for name, (method, options) in methods.items():
            h = method(fr.shape, spacing, **optics, **options, dtype=dtype)
            predictions[name] = predict_contrast(fr, fi, *h, dtype=dtype)[roi]
            metrics[name] = dict(
                versus_linear=error_metrics(predictions[name], reference["linear_contrast"]),
                versus_born=error_metrics(predictions[name], reference["born_contrast"]),
                transfer_dtype=str(h[0].dtype),
                prediction_dtype=str(predictions[name].dtype),
            )

        convergence = []
        for nz in (17, 33, 65, 129):
            fr_pad, fi_pad, _, options = make_case((nz, fr.shape[1], fr.shape[2]))
            h = wotf_ewald(fr_pad.shape, spacing, **options, axial="linear", dtype=dtype)
            start = nz // 2 - 4
            prediction = predict_contrast(fr_pad, fi_pad, *h, dtype=dtype)[start:start + 9]
            convergence.append(dict(nz=nz, dfz=1 / (nz * spacing[0]), **error_metrics(prediction, reference["linear_contrast"])))

        amplitude = []
        for scale in (1.0, 0.5, 0.25):
            scaled = born_stack(scale * fr, scale * fi, spacing, **optics, z_focus=z_focus)
            residual = scaled["born_contrast"] - scaled["linear_contrast"]
            amplitude.append(dict(
                scale=scale, quadratic_residual_l2=float(jnp.linalg.norm(residual)),
                relative_to_linear=float(jnp.linalg.norm(residual) / jnp.linalg.norm(scaled["linear_contrast"])),
            ))

        report = dict(
            dtype=dtype, reference_dtype="float64", devices=[str(d) for d in jax.devices()],
            shape=list(fr.shape), spacing_um=list(spacing), wavelength_um=optics["wavelength"],
            n_medium=optics["n_medium"], na_detection=0.6,
            source_bins_yx=[[0, 0], [1, 2], [-1, 1]], source_weights=[1, 0.6, 0.3],
            specimen="five axial layers, separate negative real/imaginary offset Gaussians; see make_case",
            pupil_phase="0.25*(fx/cutoff)^2+0.13*fy/cutoff radians",
            focus_z_um=[float(z) for z in z_focus], background=float(reference["background"]),
            max_abs_linear_contrast=float(jnp.max(jnp.abs(reference["linear_contrast"]))),
            methods=metrics, axial_convergence=convergence, amplitude_scaling=amplitude,
            boundary="periodic xy; finite z specimen; padded convolution and central focus ROI",
        )
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        _plot(output, spacing, z_focus, fr, fi, reference, predictions, convergence)
        print(json.dumps(report, indent=2))
        return report


def _plot(output, spacing, z_focus, fr, fi, reference, predictions, convergence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ref = np.asarray(reference["linear_contrast"])
    names = list(predictions)
    limit = float(np.max(np.abs(ref)))
    # Pixel edges lie half a sample beyond the centered sample coordinates.
    ny, nx = fr.shape[1:]
    extent = [(-(nx // 2) - 0.5) * spacing[2], (nx - nx // 2 - 0.5) * spacing[2],
              (-(ny // 2) - 0.5) * spacing[1], (ny - ny // 2 - 0.5) * spacing[1]]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
    fig.suptitle("Focus z = 0 um; residual panels use individual color scales", fontsize=11)
    images = [ref] + [np.asarray(predictions[name]) for name in names]
    for col, (name, stack) in enumerate(zip(["linear Born"] + names, images)):
        im = axes[0, col].imshow(stack[4], origin="lower", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[0, col].set_title(name, fontsize=11)
        axes[0, col].set(xlabel="x [um]", ylabel="y [um]")
        if col == 0:
            data, title = np.asarray(reference["born_contrast"])[4] - ref[4], "Born quadratic term"
        else:
            data, title = stack[4] - ref[4], "residual vs linear Born"
        error_limit = max(float(np.max(np.abs(data))), np.finfo(float).tiny)
        err = axes[1, col].imshow(
            data, origin="lower", extent=extent,
            cmap="viridis" if col == 0 else "RdBu_r",
            vmin=0 if col == 0 else -error_limit, vmax=error_limit,
        )
        axes[1, col].set_title(title, fontsize=10)
        axes[1, col].set(xlabel="x [um]", ylabel="y [um]")
        fig.colorbar(err, ax=axes[1, col], format="%.1e")
    fig.colorbar(im, ax=axes[0, :], label="normalized contrast")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout="constrained")
    cy, cx = fr.shape[1] // 2, fr.shape[2] // 2
    axes[0].plot(np.asarray(z_focus), ref[:, cy, cx], "k-", label="linear Born")
    for name in names:
        axes[0].plot(np.asarray(z_focus), np.asarray(predictions[name])[:, cy, cx], ".--", label=name)
    axes[0].set(xlabel="focus z [um]", ylabel="contrast", title="Axial profile")
    axes[0].legend(fontsize=7)
    axes[1].loglog([c["dfz"] for c in convergence], [c["relative_l2"] for c in convergence], "o-")
    axes[1].set_xticks([c["dfz"] for c in convergence], [f"{c['dfz']:.3f}" for c in convergence])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].set(xlabel="dfz [cycles/um]", ylabel="relative contrast L2 error", title="Linear cap gridding")
    axes[1].grid(True)
    z_specimen = (np.arange(fr.shape[0]) - fr.shape[0] // 2) * spacing[0]
    axes[2].plot(z_specimen, np.asarray(fr)[:, cy, cx], label="F_real")
    axes[2].plot(z_specimen, np.asarray(fi)[:, cy, cx], label="F_imag")
    axes[2].set(xlabel="specimen z [um]", ylabel="F [um^-2]", title="Known specimen")
    axes[2].legend()
    fig.savefig(output / "profiles.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_validation(args.dtype, args.output or Path(f"artifacts/born-{args.dtype}"))
