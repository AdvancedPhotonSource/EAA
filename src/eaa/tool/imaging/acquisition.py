from typing import Annotated, Dict, List, Any
import logging
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage as ndi
from sciagent.tool.base import BaseTool, check, ToolReturnType, tool
from sciagent.util import get_timestamp

import eaa.maths
import eaa.image_proc as ip

logger = logging.getLogger(__name__)


class AcquireImage(BaseTool):
    
    name: str = "acquire_image"
    
    @check
    def __init__(
        self,
        show_image_in_real_time: bool = False,
        *args,
        require_approval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, require_approval=require_approval, **kwargs)
        
        self.show_image_in_real_time = show_image_in_real_time
        self.rt_fig = None
        
        # Buffered images:
        # image_0 - the first image
        # image_km1 - the previous image
        # image_k - the current image
        self.image_0: np.ndarray = None
        self.image_km1: np.ndarray = None
        self.image_k: np.ndarray = None
        self.psize_0 = None
        self.psize_km1 = None
        self.psize_k = None
        
        self.image_acquisition_call_history: List[Dict[str, Any]] = []
        self.line_scan_call_history: List[Dict[str, Any]] = []
                
    @property
    def counter_acquire_image(self):
        return len(self.image_acquisition_call_history)
        
    def update_image_acquisition_call_history(
        self,
        x_center: float,
        y_center: float,
        size_x: float,
        size_y: float,
        psize_x: float,
        psize_y: float
    ):
        self.image_acquisition_call_history.append({
            "x_center": x_center,
            "y_center": y_center,
            "size_x": size_x,
            "size_y": size_y,
            "psize_x": psize_x,
            "psize_y": psize_y,
        })
        
    def update_line_scan_call_history(
        self,
        step: float,
        x_center: float,
        y_center: float,
        length: float,
        angle: float,
    ):
        self.line_scan_call_history.append({
            "step": step,
            "x_center": x_center,
            "y_center": y_center,
            "length": length,
            "angle": angle,
        })
    

    def update_real_time_view(self, image: np.ndarray):
        if self.rt_fig is None:
            self.rt_fig, ax = plt.subplots(1, 1, squeeze=True)
        else:
            ax = self.rt_fig.get_axes()[0]
        ax.clear()
        ax.imshow(image)
        plt.draw()
        plt.pause(0.001)  # Small pause to allow GUI to update
        
    def update_image_buffers(self, new_image: np.ndarray, psize: float = 1):
        """Update the image buffers.
        
        Parameters
        ----------
        new_image : np.ndarray
            The new image.
        psize : float, optional
            The pixel size (or scan step) of the new image.
        """
        if self.image_0 is None:
            self.image_0 = new_image
            self.psize_0 = psize
        self.image_km1 = self.image_k
        self.psize_km1 = self.psize_k
        self.image_k = new_image
        self.psize_k = psize

    @tool(name="acquire_image", return_type=ToolReturnType.IMAGE_PATH)
    def acquire_image(self, *args, **kwargs):
        raise NotImplementedError


class SimulatedAcquireImage(AcquireImage):
    
    name: str = "simulated_acquire_image"
    
    def __init__(
        self,
        whole_image: np.ndarray,
        return_message: bool = True,
        add_axis_ticks: bool = False,
        n_ticks: int = 10,
        add_grid_lines: bool = False,
        invert_yaxis: bool = False,
        line_scan_gaussian_fit_y_threshold: float = 0,
        add_line_scan_candidates_to_image: bool = False,
        plot_image_in_log_scale: bool = False,
        line_scan_return_gaussian_fit: bool = False,
        poisson_noise_scale: float = None,
        gaussian_psf_sigma: float = None,
        scan_jitter: float = None,
        *args,
        require_approval: bool = False,
        **kwargs
    ):
        """The simulated acquisition tool.

        Parameters
        ----------
        whole_image : np.ndarray
            A (h, w) numpy array giving the whole image. Images captured
            with specific locations and sizes will be interpolated from this
            whole image.
        return_message : bool, optional
            If True, the tool returns a string giving the path of the acquired
            image saved on the local hard drive. If False, the tool returns a
            numpy array of the acquired image.
        add_axis_ticks : bool, optional
            If True, the tool adds axis ticks to the acquired image that indicate
            the positions.
        add_grid_lines : bool, optional
            If True, the tool adds grid lines to the image.
        invert_yaxis : bool, optional
            If True, the tool inverts the y-axis of the acquired image.
        line_scan_gaussian_fit_y_threshold : float, optional
            The threshold for the Gaussian fit of the line scan. Only points whose
            y values are above y_min + y_threshold * (y_max - y_min) are considered
            for fitting. To disable point selection, set y_threshold to 0.
        add_line_scan_candidates_to_image : bool, optional
            If True, the tool adds line scan candidates to the image.
        plot_image_in_log_scale : bool, optional
            If True, 2D images are plotted in log scale.
        line_scan_return_gaussian_fit : bool, optional
            If True, the function returns a stringified JSON object containing the image path
            and the Gaussian fit FWHM.
        poisson_noise_scale : float, optional
            If given, Poisson noise is added to the sampled signal. The scaling factor
            is computed as poisson_noise_scale / max(image), which controls the effective
            photon count (and thus the noise level).
        gaussian_psf_sigma : float, optional
            If given, each sampled point is computed as a Gaussian-weighted average of
            its neighborhood within 2*sigma pixels, simulating the point spread function.
            When blur > 0 is also set, the effective sigma is scaled by (1 + blur).
        scan_jitter : float, optional
            If given, scan positions are perturbed by a value drawn uniformly from
            (-scan_jitter, scan_jitter) in both x and y.
        """
        self.whole_image = whole_image
        self.interpolator = None
        self.blur = None
        self.offset = np.array([0, 0])
        self.poisson_noise_scale = poisson_noise_scale
        self.gaussian_psf_sigma = gaussian_psf_sigma
        self.scan_jitter = scan_jitter
        self.line_scan_gaussian_fit_y_threshold = line_scan_gaussian_fit_y_threshold
        
        self.return_message = return_message
        self.add_axis_ticks = add_axis_ticks
        self.n_ticks = n_ticks
        self.add_grid_lines = add_grid_lines
        self.invert_yaxis = invert_yaxis
        self.add_line_scan_candidates_to_image = add_line_scan_candidates_to_image
        self.plot_image_in_log_scale = plot_image_in_log_scale
        self.line_scan_return_gaussian_fit = line_scan_return_gaussian_fit
        
        self.line_scan_candidates: Dict[int, list[int]] = {}

        super().__init__(*args, require_approval=require_approval, **kwargs)
        
    
    def build(self):
        self.build_interpolator()
        
    def build_interpolator(self, *args, **kwargs):
        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            (
                np.arange(self.whole_image.shape[0]),
                np.arange(self.whole_image.shape[1]),
            ),
            self.whole_image,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        self.line_interpolator = scipy.interpolate.RegularGridInterpolator(
            (
                np.arange(self.whole_image.shape[0]), 
                np.arange(self.whole_image.shape[1])
            ),
            self.whole_image,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        
    def set_blur(self, blur: float):
        """Set the amount of blurring added to the acquired image
        to simulate out of focus effect.

        Parameters
        ----------
        blur : float
            The standard deviation of the Gaussian blur.
        """
        self.blur = blur
        
    def set_offset(self, offset: np.ndarray):
        """Set the offset of the acquired image to simulate drift.

        Parameters
        ----------
        offset : np.ndarray
            The offset of the acquired image. The offset is given as a tuple
            of (y, x) coordinates.
        """
        self.offset = offset
        logger.info(f"Offset set to {self.offset}")
        
    def add_line_scan_candidates(
        self, 
        fig: plt.Figure, 
        length: float = 30,
        gap: float = 5,
        spacing: float = 30,
        horizontal: bool = True,
    ):
        """Add markers indicating line scan paths that can be chosen from
        to a figure.
        
        Parameters
        ----------
        fig : plt.Figure
            The figure to add the markers to.
        ny, nx : int
            The number of markers to add in the y and x directions.
        length : float
            The length of the markers.
        gap : float
            The gap between the ends of the markers.
        spacing : float
            The parallel spacing between the markers.
        horizontal : bool, optional
            If True, the markers are added horizontally. If False, the markers
            are added vertically.
        """
        self.line_scan_candidates = {}
        
        ax = fig.get_axes()[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ylim_sorted = sorted(ylim)
        if horizontal:
            start_xs = np.arange(xlim[0], xlim[1], length + gap)
            end_xs = start_xs + length
            start_ys = np.arange(ylim_sorted[0], ylim_sorted[1], spacing)
            end_ys = start_ys
        else:
            start_ys = np.arange(ylim_sorted[0], ylim_sorted[1], length + gap)
            end_ys = start_ys + length
            start_xs = np.arange(xlim[0], xlim[1], spacing)
            end_xs = start_xs
        start_xs_all, start_ys_all = np.meshgrid(start_xs, start_ys, indexing="ij")
        end_xs_all, end_ys_all = np.meshgrid(end_xs, end_ys, indexing="ij")
        start_xs_all = start_xs_all.flatten()
        start_ys_all = start_ys_all.flatten()
        end_xs_all = end_xs_all.flatten()
        end_ys_all = end_ys_all.flatten()
        for i in range(len(start_xs_all)):
            ax.plot([start_xs_all[i], end_xs_all[i]], [start_ys_all[i], end_ys_all[i]], color="red")
            ax.text(
                (start_xs_all[i] + end_xs_all[i]) / 2,
                (start_ys_all[i] + end_ys_all[i]) / 2,
                f"{i}",
                color="red",
                horizontalalignment="center",
                verticalalignment="bottom"
            )
            self.line_scan_candidates[i] = [start_xs_all[i], start_ys_all[i], end_xs_all[i], end_ys_all[i]]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return fig

    def _sample(self, yy_flat: np.ndarray, xx_flat: np.ndarray, shape=None) -> np.ndarray:
        """Sample the source image at the given coordinates, applying physics effects.

        Coordinates must already include any drift offset. The returned array has
        the same shape as ``yy_flat`` / ``xx_flat``.

        Parameters
        ----------
        yy_flat, xx_flat : np.ndarray
            Flat coordinate arrays with drift offset already applied.
        shape : tuple, optional
            When provided, ``gaussian_filter`` (blur fallback) is applied in this
            N-D shape rather than on the flat array, ensuring isotropic blur.
            Pass ``(size_y, size_x)`` for 2-D image acquisitions; omit for 1-D
            line scans (where blurring along the scan direction is correct).

        When none of ``poisson_noise_scale``, ``gaussian_psf_sigma``, and
        ``scan_jitter`` are set the method falls back to plain interpolation
        (plus ``gaussian_filter`` if ``blur`` is set), matching the original
        behaviour exactly.
        """
        def _apply_blur(arr_flat):
            if shape is not None:
                return ndi.gaussian_filter(arr_flat.reshape(shape), self.blur, mode="nearest").ravel()
            return ndi.gaussian_filter(arr_flat, self.blur, mode="nearest")

        use_advanced = (
            self.poisson_noise_scale is not None
            or self.gaussian_psf_sigma is not None
            or self.scan_jitter is not None
        )

        if not use_advanced:
            pts = np.column_stack([yy_flat, xx_flat])
            arr_flat = self.interpolator(pts)
            if self.blur is not None and self.blur > 0:
                arr_flat = _apply_blur(arr_flat)
            return arr_flat

        # Step 1: scan jittering
        if self.scan_jitter is not None:
            yy_flat = yy_flat + np.random.uniform(-self.scan_jitter, self.scan_jitter, yy_flat.shape)
            xx_flat = xx_flat + np.random.uniform(-self.scan_jitter, self.scan_jitter, xx_flat.shape)

        # Step 2: Gaussian PSF (or plain interpolation)
        if self.gaussian_psf_sigma is not None:
            effective_sigma = self.gaussian_psf_sigma
            if self.blur is not None and self.blur > 0:
                effective_sigma = effective_sigma * (1 + self.blur)

            half_n = max(1, int(np.ceil(2 * effective_sigma)))
            offsets = np.arange(-half_n, half_n + 1)
            dy_grid, dx_grid = np.meshgrid(offsets, offsets, indexing="ij")
            dy_flat = dy_grid.ravel()
            dx_flat = dx_grid.ravel()

            r2 = dy_flat ** 2 + dx_flat ** 2
            gauss_weights = np.exp(-0.5 * r2 / effective_sigma ** 2)
            gauss_weights /= gauss_weights.sum()

            arr_flat = np.zeros(yy_flat.size)
            for k in range(len(dy_flat)):
                nbr_pts = np.column_stack([yy_flat + dy_flat[k], xx_flat + dx_flat[k]])
                arr_flat += gauss_weights[k] * self.interpolator(nbr_pts)
        else:
            pts = np.column_stack([yy_flat, xx_flat])
            arr_flat = self.interpolator(pts)
            if self.blur is not None and self.blur > 0:
                arr_flat = _apply_blur(arr_flat)

        # Step 3: Poisson noise
        if self.poisson_noise_scale is not None:
            max_val = arr_flat.max()
            if max_val > 0:
                scaling_factor = self.poisson_noise_scale / max_val
                arr_flat = np.random.poisson(np.clip(arr_flat * scaling_factor, 0, None)).astype(float) / scaling_factor

        return arr_flat

    @tool(name="acquire_image", return_type=ToolReturnType.IMAGE_PATH)
    def acquire_image(
        self,
        y_center: Annotated[float, "The y-coordinate of the center of the image to acquire."],
        x_center: Annotated[float, "The x-coordinate of the center of the image to acquire."],
        size_y: Annotated[int, "The height of the image to acquire."],
        size_x: Annotated[int, "The width of the image to acquire."],
        scan_step: Annotated[float, "The step size between sampled points in both y and x directions."] = 1,
    ) -> Annotated[str, "The path to the acquired image."]:
        """Acquire an image of a given size from the whole image centered at a
        given location.

        Parameters
        ----------
        y_center, x_center : float
            The center of the image to acquire. The location can be a floating
            point number, in which case the image will be interpolated.
        size_y, size_x : int
            The size of the image to acquire.

        Returns
        -------
        str
            The path of the acquired image saved in hard drive.
        """
        loc_y = y_center - size_y / 2
        loc_x = x_center - size_x / 2
        self.update_image_acquisition_call_history(x_center, y_center, size_x, size_y, psize_x=scan_step, psize_y=scan_step)

        loc = [loc_y, loc_x]
        size = [size_y, size_x]
        logger.info(f"Acquiring image of size {size} centered at ({y_center}, {x_center}) with scan_step={scan_step}.")
        y = np.arange(loc[0], loc[0] + size[0], scan_step)
        x = np.arange(loc[1], loc[1] + size[1], scan_step)
        yy, xx = np.meshgrid(y + self.offset[0], x + self.offset[1], indexing="ij")

        arr_shape = (len(y), len(x))
        arr = self._sample(yy.ravel(), xx.ravel(), shape=arr_shape).reshape(arr_shape)

        if self.show_image_in_real_time:
            self.update_real_time_view(arr)

        self.update_image_buffers(arr, psize=scan_step)
            
        if self.return_message:
            filename = f"image_{y_center}_{x_center}_{size_y}_{size_x}_{get_timestamp()}.png"
            fig = ip.plot_2d_image(
                arr if not self.plot_image_in_log_scale else np.log10(arr + 1),
                add_axis_ticks=self.add_axis_ticks,
                x_coords=x,
                y_coords=y,
                n_ticks=self.n_ticks,
                add_grid_lines=self.add_grid_lines,
                invert_yaxis=self.invert_yaxis
            )
            if self.add_line_scan_candidates_to_image:
                fig = self.add_line_scan_candidates(fig)
            self.save_image_to_temp_dir(fig, filename, add_timestamp=False)
            return f".tmp/{filename}"
        else:
            return arr

    @tool(name="acquire_line_scan", return_type=ToolReturnType.IMAGE_PATH)
    def acquire_line_scan(
        self,
        x_center: Annotated[float, "The x-coordinate of the center of the line scan."],
        y_center: Annotated[float, "The y-coordinate of the center of the line scan."],
        length: Annotated[float, "The length of the line scan."],
        scan_step: float,
        angle: Annotated[float, "The angle of the line scan in degrees. 0 degrees means horizontal (along the x-axis). Positive angles rotate counter-clockwise."] = 0.0,
    ) -> Annotated[str, "The path to the plot of the line scan."]:
        """Scan along a line in the sample. This function
        generates a plot of the line scan, and a Gaussian fit
        of it. The FWHM of the Gaussian fit is annotated on the plot.

        Parameters
        ----------
        x_center, y_center : float
            The center of the line scan.
        length : float
            The length of the line scan.
        scan_step : float
            The step size of the line scan.
        angle : float
            The angle of the line scan in degrees. 0 degrees means horizontal
            (along the x-axis). Positive angles rotate counter-clockwise.

        Returns
        -------
        str
            The path of the plot of the line scan saved in hard drive.
        """
        angle_rad = np.radians(angle)
        half = length / 2
        start_x = x_center - half * np.cos(angle_rad)
        end_x = x_center + half * np.cos(angle_rad)
        start_y = y_center - half * np.sin(angle_rad)
        end_y = y_center + half * np.sin(angle_rad)
        self.update_line_scan_call_history(
            step=scan_step, x_center=x_center, y_center=y_center,
            length=length, angle=angle,
        )

        pt_start = np.array([start_y, start_x])
        pt_end = np.array([end_y, end_x])
        d_tot = np.linalg.norm(pt_end - pt_start)
        ds = np.arange(0, d_tot, scan_step)
        pts = pt_start + ds[:, None] * (pt_end - pt_start) / d_tot
        pts = pts + self.offset

        arr = self._sample(pts[:, 0], pts[:, 1])
            
        # Fit a Gaussian to the line scan
        a, mu, sigma, c, normalized_residual, fit_x_min, fit_x_max = eaa.maths.fit_gaussian_1d(
            ds, arr, y_threshold=self.line_scan_gaussian_fit_y_threshold
        )
        if np.any(np.isnan([a, mu, sigma, c])):
            val_gauss = None
            fwhm = np.nan
        else:
            val_gauss = eaa.maths.gaussian_1d(ds, a, mu, sigma, c)
            fwhm = 2.35 * np.abs(sigma)
        
        show_scan_line = self.image_k is not None and len(self.image_acquisition_call_history) > 0
        show_first_scan_line = (
            show_scan_line
            and self.image_0 is not None
            and len(self.image_acquisition_call_history) > 1
            and len(self.line_scan_call_history) > 1
        )
        if show_scan_line:
            n_cols = 3 if show_first_scan_line else 2
            fig, axes = plt.subplots(1, n_cols, squeeze=True, figsize=(5 * n_cols, 4))
            line_ax = axes[0]
            image_ax = axes[1]
            first_image_ax = axes[2] if show_first_scan_line else None
        else:
            fig, line_ax = plt.subplots(1, 1, squeeze=True)
            image_ax = None
            first_image_ax = None

        line_ax.plot(ds, arr, label="data")
        if val_gauss is not None:
            line_ax.plot(ds, val_gauss, linestyle="--", label="Gaussian fit")
        line_ax.text(
            0.05, 
            0.95, 
            f"FWHM = {fwhm:.2f}", 
            transform=line_ax.transAxes, 
            verticalalignment="top", 
            horizontalalignment="left"
        )
        line_ax.legend()
        line_ax.set_xlabel("distance")
        line_ax.set_ylabel("value")
        line_ax.set_title("Line scan")
        line_ax.grid(True)

        if show_scan_line and image_ax is not None:
            image_info = self.image_acquisition_call_history[-1]
            image_to_plot = self.image_k
            if self.plot_image_in_log_scale:
                image_to_plot = np.log10(image_to_plot + 1)
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

        if show_first_scan_line and first_image_ax is not None:
            first_image_info = self.image_acquisition_call_history[0]
            first_line_info = self.line_scan_call_history[0]
            first_image_to_plot = self.image_0
            if self.plot_image_in_log_scale:
                first_image_to_plot = np.log10(first_image_to_plot + 1)
            first_image_x_min = first_image_info["x_center"] - first_image_info["size_x"] / 2
            first_image_x_max = first_image_info["x_center"] + first_image_info["size_x"] / 2
            first_image_y_min = first_image_info["y_center"] - first_image_info["size_y"] / 2
            first_image_y_max = first_image_info["y_center"] + first_image_info["size_y"] / 2
            first_image_ax.imshow(
                first_image_to_plot,
                cmap="inferno",
                origin="upper",
                extent=[first_image_x_min, first_image_x_max, first_image_y_max, first_image_y_min],
            )
            _half = first_line_info["length"] / 2
            _angle_rad = np.radians(first_line_info["angle"])
            first_image_ax.plot(
                [
                    first_line_info["x_center"] - _half * np.cos(_angle_rad),
                    first_line_info["x_center"] + _half * np.cos(_angle_rad),
                ],
                [
                    first_line_info["y_center"] - _half * np.sin(_angle_rad),
                    first_line_info["y_center"] + _half * np.sin(_angle_rad),
                ],
                color="blue",
                linewidth=2,
            )
            first_image_ax.set_xlabel("x")
            first_image_ax.set_ylabel("y")
            first_image_ax.set_title("Reference line scan position")
            if self.invert_yaxis:
                first_image_ax.invert_yaxis()

        fig.tight_layout()
        
        if not os.path.exists(".tmp"):
            os.makedirs(".tmp")
        fname = os.path.join(
            ".tmp",
            f"line_scan_{y_center}_{x_center}_{length}_{angle}_{scan_step}_{get_timestamp()}.png"
        )
        fig.savefig(fname)
        plt.close(fig)
        if self.line_scan_return_gaussian_fit:
            return json.dumps({
                "image_path": fname,
                "fwhm": fwhm,
                "a": a,
                "mu": mu,
                "sigma": sigma,
                "c": c,
                "normalized_residual": normalized_residual,
                "x_min": fit_x_min,
                "x_max": fit_x_max,
            })
        else:
            return fname

    def scan_line_by_choice(
        self, 
        choice: Annotated[int, "The index of the line scan candidate to use."],
        scan_step: Annotated[float, "The step size of the line scan."] = 1.0,
    ) -> Annotated[str, "The path to the plot of the line scan."]:
        """Conduct a line scan along a chosen path. To use this tool,
        you must call the tool "acquire_image" first, examine the image
        with the candidates, and then call this tool with the index of the
        candidate you want to use.
        
        Parameters
        ----------
        choice : int
            The index of the line scan candidate to use. You should have
            seen an image with the line scan candidates.
        scan_step : float
            The step size of the line scan.

        Returns
        -------
        str
            The path of the plot of the line scan saved in hard drive.
        """
        start_x, start_y, end_x, end_y = self.line_scan_candidates[choice]
        dx = end_x - start_x
        dy = end_y - start_y
        x_center = (start_x + end_x) / 2
        y_center = (start_y + end_y) / 2
        length = np.sqrt(dx**2 + dy**2)
        angle = np.degrees(np.arctan2(dy, dx))
        return self.acquire_line_scan(
            x_center=x_center, y_center=y_center, length=length,
            scan_step=scan_step, angle=angle,
        )
