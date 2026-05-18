"""Simulated image-acquisition instrument implementation.

This module intentionally has no EAA dependency. It recreates the core behavior
of ``eaa_imaging.tool.imaging.acquisition.SimulatedAcquireImage`` for a process
that can own instrument-control state independently of the MCP server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Return a filesystem-friendly timestamp."""
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def gaussian_1d(x: np.ndarray, a: float, mu: float, sigma: float, c: float) -> np.ndarray:
    """Evaluate a one-dimensional Gaussian with offset."""
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c


def fit_gaussian_1d(
    x: np.ndarray,
    y: np.ndarray,
    y_threshold: float = 0.0,
) -> tuple[float, float, float, float, float, float, float]:
    """Fit a one-dimensional Gaussian curve.

    Parameters
    ----------
    x, y
        Data to fit.
    y_threshold
        Fractional threshold above ``min(y)`` used to select fit points. A value
        of zero uses all points.

    Returns
    -------
    tuple
        ``a, mu, sigma, c, normalized_residual, x_min, x_max``. Failed fits
        return NaN fit parameters.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 4 or y.size < 4:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    fit_x = x
    fit_y = y
    if y_threshold > 0:
        threshold = y.min() + y_threshold * (y.max() - y.min())
        mask = y >= threshold
        if np.count_nonzero(mask) >= 4:
            fit_x = x[mask]
            fit_y = y[mask]

    amplitude = float(fit_y.max() - fit_y.min())
    center = float(fit_x[np.argmax(fit_y)])
    sigma = float(max((fit_x.max() - fit_x.min()) / 6, np.finfo(float).eps))
    offset = float(fit_y.min())
    try:
        params, _ = optimize.curve_fit(
            gaussian_1d,
            fit_x,
            fit_y,
            p0=(amplitude, center, sigma, offset),
            maxfev=10000,
        )
        a, mu, sigma, c = (float(value) for value in params)
        residual = fit_y - gaussian_1d(fit_x, a, mu, sigma, c)
        denom = float(np.linalg.norm(fit_y)) or 1.0
        normalized_residual = float(np.linalg.norm(residual) / denom)
        return a, mu, sigma, c, normalized_residual, float(fit_x.min()), float(fit_x.max())
    except Exception:
        logger.exception("Gaussian fit failed")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, float(fit_x.min()), float(fit_x.max()))


def create_default_image(shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    """Create a deterministic synthetic image for the dummy instrument."""
    y, x = np.indices(shape, dtype=float)
    image = (
        40 * np.exp(-((x - 155) ** 2 + (y - 160) ** 2) / (2 * 42**2))
        + 95 * np.exp(-((x - 330) ** 2 + (y - 245) ** 2) / (2 * 55**2))
        + 65 * np.exp(-((x - 245) ** 2 + (y - 375) ** 2) / (2 * 35**2))
        + 12 * np.sin(x / 18)
        + 8 * np.cos(y / 23)
    )
    image -= image.min()
    return image.astype(np.float32)


@dataclass
class SimulatedInstrument:
    """Stateful simulated instrument for acquisition and parameter tuning.

    Parameters
    ----------
    whole_image
        Source image sampled by acquisition and line-scan commands.
    output_dir
        Directory where generated plots and array artifacts are written.
    add_axis_ticks
        Add physical coordinate ticks to acquired image plots.
    n_ticks
        Number of axis ticks when ``add_axis_ticks`` is enabled.
    add_grid_lines
        Add grid lines to acquired image plots.
    invert_yaxis
        Invert the y-axis on image plots.
    line_scan_gaussian_fit_y_threshold
        Fractional threshold for Gaussian line-scan fitting.
    plot_image_in_log_scale
        Plot acquired images in ``log10(image + 1)`` scale.
    line_scan_return_gaussian_fit
        Include Gaussian fit metadata in line-scan responses.
    poisson_noise_scale
        Optional Poisson noise scale.
    gaussian_psf_sigma
        Optional Gaussian point-spread-function sigma.
    scan_jitter
        Optional uniform coordinate jitter.
    parameter_names
        Names of simulated tunable parameters.
    parameter_ranges
        Lower and upper bounds for simulated tunable parameters.
    true_parameters
        Ideal parameter values used to calculate simulated blur.
    blur_factor
        Scale factor applied to normalized parameter error to set blur.
    drift_factor
        Scale factor applied to normalized parameter change to set offset.
    """

    whole_image: np.ndarray
    output_dir: Path
    add_axis_ticks: bool = False
    n_ticks: int = 10
    add_grid_lines: bool = False
    invert_yaxis: bool = False
    line_scan_gaussian_fit_y_threshold: float = 0.0
    plot_image_in_log_scale: bool = False
    line_scan_return_gaussian_fit: bool = False
    poisson_noise_scale: float | None = None
    gaussian_psf_sigma: float | None = None
    scan_jitter: float | None = None
    parameter_names: list[str] = field(default_factory=lambda: ["focus"])
    parameter_ranges: list[tuple[float, ...]] = field(
        default_factory=lambda: [(-1.0,), (1.0,)]
    )
    true_parameters: list[float] = field(default_factory=lambda: [0.0])
    blur_factor: float = 0.0
    drift_factor: float = 0.0
    add_line_scan_candidates_to_image: bool = False
    blur: float | None = None
    offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    writable_config_names = {
        "add_axis_ticks",
        "n_ticks",
        "add_grid_lines",
        "invert_yaxis",
        "line_scan_gaussian_fit_y_threshold",
        "plot_image_in_log_scale",
        "line_scan_return_gaussian_fit",
        "poisson_noise_scale",
        "gaussian_psf_sigma",
        "scan_jitter",
        "parameter_names",
        "parameter_ranges",
        "true_parameters",
        "blur_factor",
        "drift_factor",
        "add_line_scan_candidates_to_image",
    }

    def __post_init__(self) -> None:
        """Build interpolators and initialize state buffers."""
        self.output_dir = self.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_array_artifact_dir = self.output_dir / "acquisition_arrays"
        self.image_array_artifact_dir.mkdir(parents=True, exist_ok=True)
        self.array_dump_artifact_dir = self.output_dir / "array_dumps"
        self.array_dump_artifact_dir.mkdir(parents=True, exist_ok=True)
        self.interpolator = RegularGridInterpolator(
            (np.arange(self.whole_image.shape[0]), np.arange(self.whole_image.shape[1])),
            self.whole_image,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self.image_0: np.ndarray | None = None
        self.image_km1: np.ndarray | None = None
        self.image_k: np.ndarray | None = None
        self.psize_0: float | None = None
        self.psize_km1: float | None = None
        self.psize_k: float | None = None
        self.image_0_path: str | None = None
        self.image_km1_path: str | None = None
        self.image_k_path: str | None = None
        self.image_acquisition_call_history: list[dict[str, Any]] = []
        self.line_scan_call_history: list[dict[str, Any]] = []
        self.parameter_history: list[list[float]] = []
        self.parameters = [0.0 for _ in self.parameter_names]
        self.validate_parameter_configuration()

    @property
    def len_parameter_history(self) -> int:
        """Return the number of parameter-setting calls."""
        return len(self.parameter_history)

    def validate_parameter_configuration(self) -> None:
        """Validate simulated parameter metadata."""
        if len(self.true_parameters) != len(self.parameter_names):
            raise ValueError("true_parameters must match the number of parameter_names.")
        if len(self.parameter_ranges) != 2:
            raise ValueError("parameter_ranges must contain lower and upper bound rows.")
        lower_bounds, upper_bounds = self.parameter_ranges
        if (
            len(lower_bounds) != len(self.parameter_names)
            or len(upper_bounds) != len(self.parameter_names)
        ):
            raise ValueError("parameter_ranges must match the number of parameter_names.")

    @property
    def counter_acquire_image(self) -> int:
        """Return the number of image acquisitions."""
        return len(self.image_acquisition_call_history)

    def set_blur(self, blur: float | None) -> dict[str, Any]:
        """Set Gaussian blur applied to sampled data."""
        self.blur = blur
        return {"blur": self.blur}

    def set_offset(self, y_offset: float, x_offset: float) -> dict[str, Any]:
        """Set simulated drift offset in ``(y, x)`` order."""
        self.offset = np.array([y_offset, x_offset], dtype=float)
        return {"offset": self.offset.tolist()}

    def set_config(self, name: str, value: Any) -> dict[str, Any]:
        """Set a writable simulated instrument configuration value.

        Parameters
        ----------
        name
            Configuration attribute name.
        value
            JSON-serializable value to assign.

        Returns
        -------
        dict[str, Any]
            Updated name and value.
        """
        if name not in self.writable_config_names:
            raise ValueError(f"Unsupported configuration attribute: {name}")
        if name in {"parameter_names", "true_parameters"}:
            value = list(value)
        if name == "parameter_ranges":
            if len(value) != 2:
                raise ValueError("parameter_ranges must contain lower and upper bound rows.")
            value = [tuple(float(item) for item in row) for row in value]
        setattr(self, name, value)
        if name in {"parameter_names", "parameter_ranges", "true_parameters"}:
            self.validate_parameter_configuration()
            if len(self.parameters) != len(self.parameter_names):
                self.parameters = [0.0 for _ in self.parameter_names]
                self.parameter_history.clear()
        return {"name": name, "value": self.json_safe(value)}

    def set_attribute(self, name: str, value: Any) -> dict[str, Any]:
        """Alias for ``set_config`` used by the EAA MCP acquisition proxy."""
        return self.set_config(name=name, value=value)

    def json_safe(self, value: Any) -> Any:
        """Convert NumPy values and tuples into JSON-serializable objects."""
        if isinstance(value, dict):
            return {str(key): self.json_safe(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self.json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def encode_attribute_payload(self, value: Any) -> Any:
        """Encode an attribute value for JSON transfer over MCP.

        NumPy arrays are represented with dtype, shape, and nested list data so
        an adapter can reconstruct the original array without importing EAA.
        """
        if isinstance(value, np.ndarray):
            return {
                "encoding": "numpy.ndarray",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
        return self.json_safe(value)

    def get_attribute_payload(self, name: str) -> Any:
        """Return a JSON-serializable payload for a server-side attribute.

        Parameters
        ----------
        name
            Native attribute name to fetch, equivalent to ``getattr(tool,
            name)`` in in-process EAA tools.

        Returns
        -------
        object
            Literal JSON value, or a NumPy-array payload containing
            ``encoding``, ``dtype``, ``shape``, and ``data`` fields.
        """
        if not hasattr(self, name):
            raise AttributeError(f"Unknown instrument attribute: {name}")
        return self.encode_attribute_payload(getattr(self, name))

    def set_parameters(self, parameters: list[float]) -> str:
        """Set simulated instrument tuning parameters.

        Parameters
        ----------
        parameters
            Parameter values to set. The number of values must match
            ``parameter_names`` and each value must lie inside the configured
            ``parameter_ranges``.

        Returns
        -------
        str
            Human-readable update message.
        """
        if len(parameters) != len(self.parameter_names):
            raise ValueError(
                f"Expected {len(self.parameter_names)} parameter values, got {len(parameters)}."
            )
        self.validate_parameter_configuration()
        lower_bounds, upper_bounds = self.parameter_ranges
        values = [float(value) for value in parameters]
        for index, value in enumerate(values):
            lower = lower_bounds[index]
            upper = upper_bounds[index]
            if lower > upper:
                raise ValueError(
                    f"Parameter range for {self.parameter_names[index]} has lower > upper."
                )
            if not lower <= value <= upper:
                raise ValueError(
                    f"Parameter {self.parameter_names[index]}={value} is outside "
                    f"the allowed range ({lower}, {upper})."
                )

        scalers = np.asarray(upper_bounds, dtype=float) - np.asarray(lower_bounds, dtype=float)
        if np.any(np.isclose(scalers, 0.0)):
            raise ValueError("Parameter ranges must have nonzero width.")
        value_array = np.asarray(values, dtype=float)
        true_parameter_array = np.asarray(self.true_parameters, dtype=float)
        normalized_errors = (value_array - true_parameter_array) / scalers
        blur = float(np.abs(normalized_errors).sum() * self.blur_factor)
        self.set_blur(blur)

        if self.len_parameter_history > 0 and self.drift_factor > 0:
            initial_parameters = np.asarray(self.parameter_history[0], dtype=float)
            mean_delta = float(((initial_parameters - value_array) / scalers).mean())
            drift = np.ones(2) * mean_delta * self.drift_factor
            self.set_offset(y_offset=float(drift[0]), x_offset=float(drift[1]))

        self.parameters = values
        self.parameter_history.append(values)
        assignments = ", ".join(
            f"{name}={value}" for name, value in zip(self.parameter_names, values)
        )
        message = f"Set simulated parameters: {assignments}"
        logger.info(message)
        return message

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable instrument state summary."""
        return {
            "image_shape": list(self.whole_image.shape),
            "output_dir": str(self.output_dir),
            "blur": self.blur,
            "offset": self.offset.tolist(),
            "parameter_names": self.parameter_names,
            "parameter_ranges": [list(row) for row in self.parameter_ranges],
            "true_parameters": self.true_parameters,
            "blur_factor": self.blur_factor,
            "drift_factor": self.drift_factor,
            "parameters": self.parameters,
            "parameter_history": self.parameter_history,
            "counter_acquire_image": self.counter_acquire_image,
            "image_acquisition_call_history": self.image_acquisition_call_history,
            "line_scan_call_history": self.line_scan_call_history,
            "current_image_info": self.get_current_image_info(),
            "previous_image_info": self.get_previous_image_info(),
            "initial_image_info": self.get_initial_image_info(),
        }

    def save_image_array_artifact(self, image: np.ndarray) -> str:
        """Save an acquired image array to a managed ``.npy`` artifact."""
        path = self.image_array_artifact_dir / (
            f"image_{self.counter_acquire_image}_{get_timestamp()}.npy"
        )
        np.save(path, image)
        return str(path)

    def collect_referenced_image_array_paths(self) -> set[Path]:
        """Return currently referenced image array artifact paths."""
        paths = {self.image_0_path, self.image_km1_path, self.image_k_path}
        return {Path(path).expanduser().resolve() for path in paths if path is not None}

    def garbage_collect_image_array_artifacts(self) -> None:
        """Delete unreferenced managed ``.npy`` image artifacts."""
        live_paths = self.collect_referenced_image_array_paths()
        for path in self.image_array_artifact_dir.glob("*.npy"):
            resolved = path.resolve()
            if resolved not in live_paths:
                try:
                    resolved.unlink()
                except FileNotFoundError:
                    continue

    def update_image_buffers(self, new_image: np.ndarray, psize: float = 1.0) -> str:
        """Update first, previous, and current image buffers."""
        new_image_path = self.save_image_array_artifact(new_image)
        if self.image_0 is None:
            self.image_0 = new_image
            self.psize_0 = psize
            self.image_0_path = new_image_path
        self.image_km1 = self.image_k
        self.psize_km1 = self.psize_k
        self.image_km1_path = self.image_k_path
        self.image_k = new_image
        self.psize_k = psize
        self.image_k_path = new_image_path
        self.garbage_collect_image_array_artifacts()
        return new_image_path

    def get_image_buffer_info_by_name(self, buffer_name: str) -> dict[str, Any]:
        """Return path, pixel size, and shape metadata for an image buffer."""
        if buffer_name not in {"image_k", "image_km1", "image_0"}:
            raise ValueError("buffer_name must be one of image_k, image_km1, or image_0.")
        image = getattr(self, buffer_name)
        path = getattr(self, f"{buffer_name}_path")
        psize = getattr(self, f"psize_{buffer_name.split('_', 1)[1]}")
        return {
            "buffer_name": buffer_name,
            "path": path,
            "psize": psize,
            "shape": None if image is None else list(image.shape),
            "dtype": None if image is None else str(image.dtype),
        }

    def get_current_image_info(self) -> dict[str, Any]:
        """Return metadata for the current image buffer."""
        return self.get_image_buffer_info_by_name("image_k")

    def get_previous_image_info(self) -> dict[str, Any]:
        """Return metadata for the previous image buffer."""
        return self.get_image_buffer_info_by_name("image_km1")

    def get_initial_image_info(self) -> dict[str, Any]:
        """Return metadata for the initial image buffer."""
        return self.get_image_buffer_info_by_name("image_0")

    def dump_array(self, buffer_name: str) -> dict[str, Any]:
        """Write an image buffer to a ``.npy`` artifact and return its path.

        Parameters
        ----------
        buffer_name
            One of ``image_k``, ``image_km1``, or ``image_0``.

        Returns
        -------
        dict[str, Any]
            Artifact metadata with ``path``, ``buffer_name``, ``shape``,
            ``dtype``, and ``psize`` fields.
        """
        if buffer_name not in {"image_k", "image_km1", "image_0"}:
            raise ValueError("buffer_name must be one of image_k, image_km1, or image_0.")
        image = getattr(self, buffer_name)
        if image is None:
            raise ValueError(f"Image buffer {buffer_name!r} is empty.")
        path = self.array_dump_artifact_dir / f"{buffer_name}_{get_timestamp()}.npy"
        np.save(path, image)
        psize = getattr(self, f"psize_{buffer_name.split('_', 1)[1]}")
        return {
            "buffer_name": buffer_name,
            "path": str(path),
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "psize": psize,
        }

    def sample(self, yy_flat: np.ndarray, xx_flat: np.ndarray, shape=None) -> np.ndarray:
        """Sample the source image and apply optional simulated physics effects."""

        def apply_blur(arr_flat: np.ndarray) -> np.ndarray:
            if shape is not None:
                return ndi.gaussian_filter(arr_flat.reshape(shape), self.blur, mode="nearest").ravel()
            return ndi.gaussian_filter(arr_flat, self.blur, mode="nearest")

        use_advanced = (
            self.poisson_noise_scale is not None
            or self.gaussian_psf_sigma is not None
            or self.scan_jitter is not None
        )
        if not use_advanced:
            arr_flat = self.interpolator(np.column_stack([yy_flat, xx_flat]))
            if self.blur is not None and self.blur > 0:
                arr_flat = apply_blur(arr_flat)
            return arr_flat

        if self.scan_jitter is not None:
            yy_flat = yy_flat + np.random.uniform(-self.scan_jitter, self.scan_jitter, yy_flat.shape)
            xx_flat = xx_flat + np.random.uniform(-self.scan_jitter, self.scan_jitter, xx_flat.shape)

        if self.gaussian_psf_sigma is not None:
            effective_sigma = self.gaussian_psf_sigma
            if self.blur is not None and self.blur > 0:
                effective_sigma *= 1 + self.blur
            half_n = max(1, int(np.ceil(2 * effective_sigma)))
            offsets = np.arange(-half_n, half_n + 1)
            dy_grid, dx_grid = np.meshgrid(offsets, offsets, indexing="ij")
            dy_flat = dy_grid.ravel()
            dx_flat = dx_grid.ravel()
            weights = np.exp(-0.5 * (dy_flat**2 + dx_flat**2) / effective_sigma**2)
            weights /= weights.sum()
            arr_flat = np.zeros(yy_flat.size)
            for index, weight in enumerate(weights):
                points = np.column_stack([yy_flat + dy_flat[index], xx_flat + dx_flat[index]])
                arr_flat += weight * self.interpolator(points)
        else:
            arr_flat = self.interpolator(np.column_stack([yy_flat, xx_flat]))
            if self.blur is not None and self.blur > 0:
                arr_flat = apply_blur(arr_flat)

        if self.poisson_noise_scale is not None:
            max_val = arr_flat.max()
            if max_val > 0:
                scaling_factor = self.poisson_noise_scale / max_val
                arr_flat = (
                    np.random.poisson(np.clip(arr_flat * scaling_factor, 0, None)).astype(float)
                    / scaling_factor
                )
        return arr_flat

    def acquire_image(
        self,
        x_center: float,
        y_center: float,
        size_y: int,
        size_x: int,
        scan_step: float = 1.0,
    ) -> dict[str, Any]:
        """Acquire and plot an image sampled from the whole image.

        Coordinates are in ``x_center``, ``y_center`` order. The returned
        ``img_path`` points to a PNG display artifact and ``psize`` is the
        pixel size in the same coordinate units as the input positions. Use
        ``dump_array`` to export numerical image data from worker buffers.
        """
        loc_y = y_center - size_y / 2
        loc_x = x_center - size_x / 2
        self.image_acquisition_call_history.append(
            {
                "x_center": x_center,
                "y_center": y_center,
                "size_x": size_x,
                "size_y": size_y,
                "psize_x": scan_step,
                "psize_y": scan_step,
            }
        )
        y = np.arange(loc_y, loc_y + size_y, scan_step)
        x = np.arange(loc_x, loc_x + size_x, scan_step)
        yy, xx = np.meshgrid(y + self.offset[0], x + self.offset[1], indexing="ij")
        arr_shape = (len(y), len(x))
        arr = self.sample(yy.ravel(), xx.ravel(), shape=arr_shape).reshape(arr_shape)
        self.update_image_buffers(arr, psize=scan_step)

        image_to_plot = arr if not self.plot_image_in_log_scale else np.log10(arr + 1)
        fig, ax = plt.subplots()
        ax.imshow(image_to_plot, cmap="inferno", origin="upper")
        if self.add_axis_ticks:
            x_ticks = np.linspace(0, max(len(x) - 1, 0), min(self.n_ticks, len(x)), dtype=int)
            y_ticks = np.linspace(0, max(len(y) - 1, 0), min(self.n_ticks, len(y)), dtype=int)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f"{x[idx]:.1f}" for idx in x_ticks])
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{y[idx]:.1f}" for idx in y_ticks])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        if self.add_grid_lines:
            ax.grid(True)
        if self.invert_yaxis:
            ax.invert_yaxis()
        fig.tight_layout()

        filename = f"image_{y_center}_{x_center}_{size_y}_{size_x}_{get_timestamp()}.png"
        path = self.output_dir / filename
        fig.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return {"img_path": str(path), "psize": scan_step}

    def acquire_line_scan(
        self,
        x_center: float,
        y_center: float,
        length: float,
        scan_step: float,
        angle: float = 0.0,
    ) -> dict[str, Any]:
        """Acquire and plot a line scan sampled from the whole image."""
        angle_rad = np.radians(angle)
        half = length / 2
        start_x = x_center - half * np.cos(angle_rad)
        end_x = x_center + half * np.cos(angle_rad)
        start_y = y_center - half * np.sin(angle_rad)
        end_y = y_center + half * np.sin(angle_rad)
        self.line_scan_call_history.append(
            {
                "step": scan_step,
                "x_center": x_center,
                "y_center": y_center,
                "length": length,
                "angle": angle,
            }
        )

        pt_start = np.array([start_y, start_x])
        pt_end = np.array([end_y, end_x])
        distance_total = np.linalg.norm(pt_end - pt_start)
        distances = np.arange(0, distance_total, scan_step)
        points = pt_start + distances[:, None] * (pt_end - pt_start) / distance_total
        points = points + self.offset
        arr = self.sample(points[:, 0], points[:, 1])

        a, mu, sigma, c, residual, fit_x_min, fit_x_max = fit_gaussian_1d(
            distances,
            arr,
            y_threshold=self.line_scan_gaussian_fit_y_threshold,
        )
        if np.any(np.isnan([a, mu, sigma, c])):
            gaussian_values = None
            fwhm = np.nan
        else:
            gaussian_values = gaussian_1d(distances, a, mu, sigma, c)
            fwhm = 2.35 * abs(sigma)

        show_scan_line = self.image_k is not None and len(self.image_acquisition_call_history) > 0
        if show_scan_line:
            fig, axes = plt.subplots(1, 2, squeeze=True, figsize=(10, 4))
            line_ax = axes[0]
            image_ax = axes[1]
        else:
            fig, line_ax = plt.subplots(1, 1, squeeze=True)
            image_ax = None

        line_ax.plot(distances, arr, label="data")
        if gaussian_values is not None:
            line_ax.plot(distances, gaussian_values, linestyle="--", label="Gaussian fit")
        line_ax.text(
            0.05,
            0.95,
            f"FWHM = {fwhm:.2f}",
            transform=line_ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
        )
        line_ax.legend()
        line_ax.set_xlabel("distance")
        line_ax.set_ylabel("value")
        line_ax.set_title("Line scan")
        line_ax.grid(True)

        if image_ax is not None and self.image_k is not None:
            image_info = self.image_acquisition_call_history[-1]
            image_to_plot = self.image_k if not self.plot_image_in_log_scale else np.log10(self.image_k + 1)
            image_x_min = image_info["x_center"] - image_info["size_x"] / 2
            image_x_max = image_info["x_center"] + image_info["size_x"] / 2
            image_y_min = image_info["y_center"] - image_info["size_y"] / 2
            image_y_max = image_info["y_center"] + image_info["size_y"] / 2
            image_ax.imshow(
                image_to_plot,
                cmap="inferno",
                origin="upper",
                extent=[image_x_min, image_x_max, image_y_max, image_y_min],
            )
            image_ax.plot([start_x, end_x], [start_y, end_y], color="red", linewidth=2)
            image_ax.set_xlabel("x")
            image_ax.set_ylabel("y")
            image_ax.set_title("Line scan position")
            if self.invert_yaxis:
                image_ax.invert_yaxis()

        fig.tight_layout()
        path = self.output_dir / f"line_scan_{y_center}_{x_center}_{length}_{angle}_{scan_step}_{get_timestamp()}.png"
        fig.savefig(path)
        plt.close(fig)

        result: dict[str, Any] = {"img_path": str(path), "fwhm": fwhm}
        if self.line_scan_return_gaussian_fit:
            result.update(
                {
                    "fwhm": fwhm,
                    "a": a,
                    "mu": mu,
                    "sigma": sigma,
                    "c": c,
                    "normalized_residual": residual,
                    "x_min": fit_x_min,
                    "x_max": fit_x_max,
                }
            )
        return result
