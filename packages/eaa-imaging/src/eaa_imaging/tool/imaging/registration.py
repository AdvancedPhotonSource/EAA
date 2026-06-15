from typing import Annotated, Any, Literal, Optional, Tuple

import eaa_core.matplotlib_setup  # noqa: F401
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from eaa_core.tool.base import BaseTool, check, tool

from eaa_imaging.image_proc import (
    error_minimization_registration,
    phase_cross_correlation,
    translation_nmi_registration,
)

class ImageRegistration(BaseTool):
    """Register image arrays supplied explicitly by callers."""

    name: str = "image_registration"

    @check
    def __init__(
        self,
        reference_image: np.ndarray = None,
        reference_pixel_size: float = 1.0,
        image_coordinates_origin: Literal["top_left", "center"] = "top_left",
        registration_method: Literal[
            "phase_correlation", "sift", "mutual_information", "error_minimization", "ncc"
        ] = "phase_correlation",
        registration_algorithm_kwargs: Optional[dict[str, Any]] = None,
        zoom: float = 1.0,
        log_scale: bool = False,
        require_approval: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the image registration tool.

        Parameters
        ----------
        reference_image : np.ndarray, optional
            The reference image to register the latest image with.
        reference_pixel_size : float, optional
            The pixel size of the reference image.
        image_coordinates_origin : Literal["top_left", "center"], optional
            The origin of the image coordinates. Useful to handle cases where
            the size of the registered images do not match. Image registration
            finds the translation to apply to the test image so that it aligns
            with the reference image where the values are the same for the same coordinates
            (i, j). When the origin defining the coordinates is different, the
            test image is cropped/padded differently when its size does not match.
            When this argument is set to "center", the test image is padded/cropped
            centrally. When it is set to "top_left", the test image is on the bottom
            and right sides.
        registration_method : Literal["phase_correlation", "sift", "mutual_information", "error_minimization", "ncc"], optional
            The method used to estimate translational offsets. "phase_correlation"
            uses phase correlation, "sift" uses feature matching,
            "mutual_information" uses pyramid-based normalized mutual information,
            "error_minimization" uses exhaustive integer-shift MSE search with
            local quadratic subpixel refinement, and "ncc" uses a tuned
            normalized-cross-correlation ensemble with subpixel refinement.
        registration_algorithm_kwargs : Optional[dict[str, Any]], optional
            Keyword arguments to pass to the registration algorithm.
        zoom : float, optional
            Zoom factor applied to both images before registration. Returned
            offsets are scaled back to the original image coordinates.
        log_scale : bool, optional
            If True, images are transformed as `log10(x + 1)` before registration.
        """
        super().__init__(*args, require_approval=require_approval, **kwargs)

        self.reference_image = reference_image
        self.reference_pixel_size = reference_pixel_size
        self.image_coordinates_origin = image_coordinates_origin
        self.registration_method = registration_method
        self.registration_algorithm_kwargs = registration_algorithm_kwargs or {}
        self.zoom = zoom
        self.log_scale = log_scale

    def set_reference_image(
        self,
        reference_image: np.ndarray,
        reference_pixel_size: float = 1.0,
    ) -> None:
        """Set the reference image."""
        self.reference_image = reference_image
        self.reference_pixel_size = reference_pixel_size

    def set_reference_image_from_path(
        self,
        reference_image_path: str,
        reference_pixel_size: float = 1.0,
    ) -> None:
        """Set the reference image from an array file.

        Parameters
        ----------
        reference_image_path : str
            Path to the reference image array.
        reference_pixel_size : float, optional
            Pixel size of the reference image.
        """
        self.set_reference_image(
            np.load(reference_image_path),
            reference_pixel_size=reference_pixel_size,
        )
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image to prepare it for registration.
        If the input image is a 3D array, the last dimension is assumed to
        be the channel dimension and will be averaged over.
        
        Parameters
        ----------
        image : np.ndarray
            A 2D or 3D array representing an image.

        Returns
        -------
        image : np.ndarray
            A 2D array representing an image.
        """
        if image.ndim == 3:
            image = np.mean(image, axis=-1)
        image[np.isnan(image)] = np.mean(image)
        if self.log_scale:
            image = np.log10(image + 1)
        return image

    def zoom_image(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured registration zoom factor to an image."""
        if self.zoom <= 0:
            raise ValueError("zoom must be positive.")
        if self.zoom == 1.0:
            return image
        return ndi.zoom(image, zoom=self.zoom, order=1, mode="nearest")

    @tool(name="get_offset_from_paths")
    def get_offset_from_paths(
        self,
        current_image_path: Annotated[str, "Path to the current image array file."],
        reference_image_path: Annotated[str, "Path to the reference image array file."],
        current_pixel_size: Annotated[float, "Pixel size of the current image."] = 1.0,
        reference_pixel_size: Annotated[float, "Pixel size of the reference image."] = 1.0,
    ) -> Annotated[
        list[float],
        "The translational offset [dy, dx] to apply to the current image "
        "so it aligns with the reference image.",
    ]:
        """Register two image arrays loaded from paths.

        Parameters
        ----------
        current_image_path : str
            Path to the current/test image array.
        reference_image_path : str
            Path to the reference image array.
        current_pixel_size : float, optional
            Pixel size of the current/test image.
        reference_pixel_size : float, optional
            Pixel size of the reference image.

        Returns
        -------
        list[float]
            Translational offset ``[dy, dx]`` in physical units.
        """
        current_image = np.load(current_image_path, allow_pickle=False)
        reference_image = np.load(reference_image_path, allow_pickle=False)
        offset = self.register_images(
            image_t=current_image,
            image_r=reference_image,
            psize_t=float(current_pixel_size),
            psize_r=float(reference_pixel_size),
            registration_algorithm_kwargs=self.registration_algorithm_kwargs,
        )
        return np.array(offset, dtype=float).tolist()

    def build_registration_pair_figure(
        self,
        image_r: np.ndarray,
        image_t: np.ndarray,
        images_are_processed: bool = False,
    ):
        if not images_are_processed:
            image_r = self.process_image(image_r)
            image_t = self.process_image(image_t)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_r, cmap="inferno", origin="upper")
        axes[0].set_title("Previous / Reference")
        axes[0].set_axis_off()
        axes[1].imshow(image_t, cmap="inferno", origin="upper")
        axes[1].set_title("Current / Target")
        axes[1].set_axis_off()
        fig.tight_layout()
        return fig

    @staticmethod
    def _ncc_preprocess(img: np.ndarray, name: str, sigma: float) -> np.ndarray:
        """Apply one of the tuned NCC preprocessing transforms."""
        if name == "log1p":
            out = np.log1p(np.clip(img, 0, None))
        elif name == "sqrt":
            out = np.sqrt(np.clip(img, 0, None))
        elif name == "rank":
            flat = img.ravel()
            ranks = np.empty_like(flat, dtype=np.float64)
            ranks[np.argsort(flat)] = np.arange(len(flat), dtype=np.float64)
            out = ranks.reshape(img.shape)
        elif name == "power":
            out = np.power(np.clip(img, 0, None) + 1e-12, 0.3)
        elif name == "asinh":
            out = np.arcsinh(img)
        else:
            raise ValueError(f"Unknown NCC preprocessing method: {name}")
        if sigma > 0:
            out = ndi.gaussian_filter(out, sigma=sigma)
        return out

    @staticmethod
    def _overlap_ncc(ref: np.ndarray, moving: np.ndarray, dy: int, dx: int) -> float:
        h, w = ref.shape
        if dy >= 0:
            ry, my = slice(dy, h), slice(0, h - dy)
        else:
            ry, my = slice(0, h + dy), slice(-dy, h)
        if dx >= 0:
            rx, mx = slice(dx, w), slice(0, w - dx)
        else:
            rx, mx = slice(0, w + dx), slice(-dx, w)
        ref_overlap = ref[ry, rx].ravel()
        moving_overlap = moving[my, mx].ravel()
        if len(ref_overlap) < 3:
            return -2.0
        ref_overlap = ref_overlap - np.mean(ref_overlap)
        moving_overlap = moving_overlap - np.mean(moving_overlap)
        denom = np.sqrt(np.sum(ref_overlap**2) * np.sum(moving_overlap**2))
        if denom <= 1e-12:
            return 0.0
        return float(np.sum(ref_overlap * moving_overlap) / denom)

    def _compute_ncc_surface(
        self,
        ref: np.ndarray,
        moving: np.ndarray,
        max_shift: int,
        min_overlap: int = 3,
    ) -> tuple[int, int, float, dict[tuple[int, int], float]]:
        h, w = ref.shape
        sy = min(max_shift, h - min_overlap)
        sx = min(max_shift, w - min_overlap)
        best_ncc, best_dy, best_dx = -2.0, 0, 0
        ncc_map: dict[tuple[int, int], float] = {}
        for dy in range(-sy, sy + 1):
            for dx in range(-sx, sx + 1):
                ncc = self._overlap_ncc(ref, moving, dy, dx)
                ncc_map[(dy, dx)] = ncc
                if ncc > best_ncc:
                    best_ncc, best_dy, best_dx = ncc, dy, dx
        return best_dy, best_dx, best_ncc, ncc_map

    @staticmethod
    def _extract_ncc_surface_patch(
        ncc_map: dict[tuple[int, int], float],
        peak_dy: int,
        peak_dx: int,
        radius: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        coords = []
        vals = []
        for ody in range(-radius, radius + 1):
            for odx in range(-radius, radius + 1):
                key = (peak_dy + ody, peak_dx + odx)
                if key in ncc_map and ncc_map[key] > -1.5:
                    coords.append([float(ody), float(odx)])
                    vals.append(ncc_map[key])
        if not coords:
            return None, None
        return np.array(coords), np.array(vals)

    @staticmethod
    def _gp_subpixel(
        coords: np.ndarray | None,
        vals: np.ndarray | None,
        step: float = 0.02,
        clamp: float = 0.5,
        lengthscale: float = 1.2,
        noise_var: float = 1e-4,
    ) -> tuple[float, float, float]:
        if coords is None or vals is None or len(coords) < 5:
            return 0.0, 0.0, -1.0
        dist_sq = np.sum(
            (coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        kernel = np.exp(-0.5 * dist_sq / (lengthscale**2))
        kernel += noise_var * np.eye(len(coords))
        try:
            chol = np.linalg.cholesky(kernel)
            alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, vals))
        except np.linalg.LinAlgError:
            kernel += 1e-3 * np.eye(len(coords))
            try:
                alpha = np.linalg.solve(kernel, vals)
            except np.linalg.LinAlgError:
                return 0.0, 0.0, -1.0

        grid_1d = np.arange(-clamp, clamp + step / 2, step)
        gy, gx = np.meshgrid(grid_1d, grid_1d, indexing="ij")
        test_pts = np.column_stack([gy.ravel(), gx.ravel()])
        dist_sq_test = np.sum(
            (test_pts[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        pred = np.exp(-0.5 * dist_sq_test / (lengthscale**2)) @ alpha
        best_idx = np.argmax(pred)
        return float(test_pts[best_idx, 0]), float(test_pts[best_idx, 1]), float(pred[best_idx])

    @staticmethod
    def _quadratic_2d_subpixel(
        coords: np.ndarray | None,
        vals: np.ndarray | None,
    ) -> tuple[float, float]:
        if coords is None or vals is None or len(coords) < 6:
            return 0.0, 0.0
        mask = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2) <= 2.0
        if mask.sum() < 6:
            mask = np.ones(len(coords), dtype=bool)
        c = coords[mask]
        v = vals[mask]
        design = np.column_stack(
            [c[:, 0] ** 2, c[:, 1] ** 2, c[:, 0] * c[:, 1], c[:, 0], c[:, 1], np.ones(len(c))]
        )
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design, v, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0
        a, b, cross, d, e, _ = coeffs
        denom = 4 * a * b - cross**2
        if abs(denom) < 1e-14 or a > 0 or b > 0:
            return 0.0, 0.0
        peak_dy = float(np.clip((cross * e - 2 * b * d) / denom, -0.5, 0.5))
        peak_dx = float(np.clip((cross * d - 2 * a * e) / denom, -0.5, 0.5))
        return peak_dy, peak_dx

    def ncc_registration(
        self,
        moving: np.ndarray,
        ref: np.ndarray,
        max_shift: int = 7,
        surface_radius: int = 3,
        gp_lengthscale: float = 1.2,
        ncc_power: float = 48.0,
        configs: Optional[list[tuple[str, float]]] = None,
    ) -> np.ndarray:
        """Register images with the tuned NCC ensemble from CVEvolve.

        Parameters
        ----------
        moving : np.ndarray
            Moving image to shift.
        ref : np.ndarray
            Reference image.
        max_shift : int, optional
            Maximum integer shift, in pixels, searched along each axis.
        surface_radius : int, optional
            Radius around the integer peak used for subpixel surface fitting.
        gp_lengthscale : float, optional
            RBF-kernel length scale for Gaussian-process subpixel refinement.
        ncc_power : float, optional
            Power applied to NCC scores when weighting preprocessing pipelines.
        configs : list[tuple[str, float]], optional
            Preprocessing pipeline names and Gaussian smoothing sigmas.

        Returns
        -------
        np.ndarray
            Estimated ``[dy, dx]`` shift to apply to ``moving``.
        """
        if configs is None:
            configs = [
                ("log1p", 0.3),
                ("log1p", 0.5),
                ("log1p", 0.7),
                ("sqrt", 0.3),
                ("sqrt", 0.5),
                ("rank", 0.3),
                ("rank", 0.5),
                ("power", 0.5),
                ("asinh", 0.5),
            ]

        votes: dict[tuple[int, int], int] = {}
        pipelines: list[dict[str, Any]] = []
        for preprocess_name, sigma in configs:
            ref_processed = self._ncc_preprocess(ref.copy(), preprocess_name, sigma)
            moving_processed = self._ncc_preprocess(moving.copy(), preprocess_name, sigma)
            peak_dy, peak_dx, peak_ncc, ncc_map = self._compute_ncc_surface(
                ref_processed,
                moving_processed,
                max_shift=max_shift,
            )
            key = (peak_dy, peak_dx)
            votes[key] = votes.get(key, 0) + 1
            pipelines.append(
                {
                    "int_dy": peak_dy,
                    "int_dx": peak_dx,
                    "peak_ncc": peak_ncc,
                    "ncc_map": ncc_map,
                }
            )

        if not pipelines:
            return np.zeros(2)

        sorted_keys = sorted(votes.keys(), key=lambda k: votes[k], reverse=True)
        best_key = sorted_keys[0]
        if len(sorted_keys) > 1 and votes[sorted_keys[1]] >= votes[best_key] - 1:
            key2 = sorted_keys[1]
            ncc1 = np.mean(
                [
                    p["peak_ncc"]
                    for p in pipelines
                    if (p["int_dy"], p["int_dx"]) == best_key
                ]
            )
            ncc2 = np.mean(
                [
                    p["peak_ncc"]
                    for p in pipelines
                    if (p["int_dy"], p["int_dx"]) == key2
                ]
            )
            if ncc2 > ncc1:
                best_key = key2
        center_dy, center_dx = best_key
        agreeing = [
            p
            for p in pipelines
            if (p["int_dy"], p["int_dx"]) == (center_dy, center_dx)
        ] or pipelines

        pipe_weights = [max(0.01, p["peak_ncc"]) ** ncc_power for p in agreeing]
        weight_total = sum(pipe_weights)
        if weight_total <= 0:
            pipe_weights = [1.0 / len(agreeing)] * len(agreeing)
        else:
            pipe_weights = [w / weight_total for w in pipe_weights]

        coord_values: dict[tuple[int, int], list[float]] = {}
        for pipeline, weight in zip(agreeing, pipe_weights):
            for ody in range(-surface_radius, surface_radius + 1):
                for odx in range(-surface_radius, surface_radius + 1):
                    key = (center_dy + ody, center_dx + odx)
                    if key in pipeline["ncc_map"] and pipeline["ncc_map"][key] > -1.5:
                        point = (ody, odx)
                        if point not in coord_values:
                            coord_values[point] = [0.0, 0.0]
                        coord_values[point][0] += weight * pipeline["ncc_map"][key]
                        coord_values[point][1] += weight

        avg_coords = []
        avg_vals = []
        for (ody, odx), (weighted_sum, total_weight) in coord_values.items():
            avg_coords.append([float(ody), float(odx)])
            avg_vals.append(float(weighted_sum / total_weight))
        avg_coords_arr = np.array(avg_coords) if avg_coords else None
        avg_vals_arr = np.array(avg_vals) if avg_vals else None

        avg_gp_dy, avg_gp_dx, _ = self._gp_subpixel(
            avg_coords_arr,
            avg_vals_arr,
            step=0.01,
            lengthscale=gp_lengthscale,
        )
        quad_dy, quad_dx = self._quadratic_2d_subpixel(avg_coords_arr, avg_vals_arr)

        per_pipe_dys = []
        per_pipe_dxs = []
        for pipeline in agreeing:
            coords, vals = self._extract_ncc_surface_patch(
                pipeline["ncc_map"],
                center_dy,
                center_dx,
                surface_radius,
            )
            if coords is None or len(coords) < 5:
                continue
            pipe_dy, pipe_dx, _ = self._gp_subpixel(
                coords,
                vals,
                step=0.02,
                lengthscale=gp_lengthscale,
            )
            if abs(pipe_dy) <= 0.55 and abs(pipe_dx) <= 0.55:
                per_pipe_dys.append(pipe_dy)
                per_pipe_dxs.append(pipe_dx)

        estimates_dy = []
        estimates_dx = []
        if abs(avg_gp_dy) <= 0.55 and abs(avg_gp_dx) <= 0.55:
            estimates_dy.append(avg_gp_dy)
            estimates_dx.append(avg_gp_dx)
        if per_pipe_dys:
            estimates_dy.append(float(np.median(per_pipe_dys)))
            estimates_dx.append(float(np.median(per_pipe_dxs)))
        if not estimates_dy:
            if abs(quad_dy) <= 0.55 and abs(quad_dx) <= 0.55:
                return np.array([center_dy + quad_dy, center_dx + quad_dx])
            return np.array([float(center_dy), float(center_dx)])

        return np.array(
            [
                center_dy + float(np.mean(estimates_dy)),
                center_dx + float(np.mean(estimates_dx)),
            ]
        )

    def register_images(
        self,
        image_t: np.ndarray,
        image_r: np.ndarray,
        psize_t: float,
        psize_r: float,
        registration_method: Optional[Literal["phase_correlation", "sift", "mutual_information", "error_minimization", "ncc"]] = None,
        registration_algorithm_kwargs: Optional[dict[str, Any]] = None,
    ) -> np.ndarray | Tuple[np.ndarray, float] | str:
        """
        Register the target image with the reference image.
        
        Parameters
        ----------
        image_t : np.ndarray
            The target image.
        image_r : np.ndarray
            The reference image.
        psize_t : float
            The pixel size of the target image.
        psize_r : float
            The pixel size of the reference image.
        registration_method : Optional[Literal["phase_correlation", "sift", "mutual_information", "error_minimization", "ncc"]], optional
            Overrides the default registration method for this call. 
        registration_algorithm_kwargs : Optional[dict[str, Any]], optional
            Optional keyword arguments forwarded to the selected registration
            algorithm. Supported keys and defaults depend on `registration_method`:

            - `registration_method="phase_correlation"`:
              - `use_hanning_window` (bool, default: `True`)
              - `upsample_factor` (int, default: `1`)

            - `registration_method="mutual_information"`:
              - `pyramid_levels` (tuple[int, ...], default: `(4, 2, 1)`)
              - `bins` (int, default: `64`)
              - `sample_frac` (float, default: `0.2`)
              - `smooth_sigmas` (Optional[dict[int, float]], default: `None`)
              - `optimizer` (Literal["powell", "nelder-mead"], default: `"nelder-mead"`)
              - `max_iter` (int, default: `60`)
              - `tol` (float, default: `1e-4`)

            - `registration_method="error_minimization"`:
              - `y_valid_fraction` (float, default: `0.8`)
              - `x_valid_fraction` (float, default: `0.8`)
              - `subpixel` (bool, default: `True`)

            - `registration_method="ncc"`:
              - `max_shift` (int, default: `7`)
              - `surface_radius` (int, default: `3`)
              - `gp_lengthscale` (float, default: `1.2`)
              - `ncc_power` (float, default: `48.0`)
              - `configs` (list[tuple[str, float]], optional)

            - `registration_method="sift"`:
              - No algorithm kwargs are currently supported; pass `None` or `{}`.

        Returns
        -------
        np.ndarray | str
            The translation offset (dy, dx) to apply to the target image so it aligns
            with the reference image. Positive y means shifting the test image
            downward; positive x means shifting the target image rightward. Returned
            values are in physical units, i.e., pixel size is already accounted for.
        """
        method = registration_method or self.registration_method
        algorithm_kwargs = dict(registration_algorithm_kwargs or {})
        image_t = self.process_image(np.array(image_t, copy=True))
        image_r = self.process_image(np.array(image_r, copy=True))

        # Handle pixel size and image size differences
        if psize_t != psize_r:
            # Resize the target image to have the same pixel size as the reference image
            image_t = ndi.zoom(image_t, psize_t / psize_r)

        image_t = self.zoom_image(image_t)
        image_r = self.zoom_image(image_r)

        if method in {"phase_correlation", "mutual_information", "error_minimization", "ncc"}:
            image_t = self.reconcile_image_shape(image_t, image_r.shape)

        if method == "phase_correlation":
            phase_kwargs = {"filtering_method": "hanning"}
            phase_kwargs.update(algorithm_kwargs)
            offset = phase_cross_correlation(
                image_t,
                image_r,
                **phase_kwargs,
            )
        elif method == "mutual_information":
            mi_kwargs = {
                "pyramid_levels": (4, 2, 1),
                "bins": 64,
                "sample_frac": 0.2,
                "optimizer": "nelder-mead",
                "max_iter": 60,
                "tol": 1e-4,
            }
            mi_kwargs.update(algorithm_kwargs)
            offset = translation_nmi_registration(
                moving=image_t,
                ref=image_r,
                **mi_kwargs,
            )
        elif method == "error_minimization":
            em_kwargs = {"y_valid_fraction": 0.8, "x_valid_fraction": 0.8, "subpixel": True}
            em_kwargs.update(algorithm_kwargs)
            offset = error_minimization_registration(image_t, image_r, **em_kwargs)
        elif method == "ncc":
            ncc_kwargs = {
                "max_shift": 7,
                "surface_radius": 3,
                "gp_lengthscale": 1.2,
                "ncc_power": 48.0,
            }
            ncc_kwargs.update(algorithm_kwargs)
            offset = self.ncc_registration(image_t, image_r, **ncc_kwargs)
        elif method == "sift":
            if len(algorithm_kwargs) > 0:
                raise ValueError(
                    "`registration_algorithm_kwargs` is not supported for "
                    "registration_method='sift'."
                )
            offset = self.feature_based_registration(image_t, image_r)
        else:
            raise ValueError(f"Invalid registration method: {method}")
        return np.array(offset, dtype=float) / self.zoom

    def reconcile_image_shape(
        self,
        image_t: np.ndarray,
        reference_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Crop or pad target image to match reference shape."""
        if image_t.shape == reference_shape:
            return image_t

        output = image_t
        for i in range(2):
            if self.image_coordinates_origin == "top_left":
                if output.shape[i] < reference_shape[i]:
                    pad_len = [(0, 0), (0, 0)]
                    pad_len[i] = (0, reference_shape[i] - output.shape[i])
                    output = np.pad(output, pad_len, mode="constant")
                elif output.shape[i] > reference_shape[i]:
                    slicer = [slice(None)] * 2
                    slicer[i] = slice(0, reference_shape[i])
                    output = output[tuple(slicer)]
            elif self.image_coordinates_origin == "center":
                if output.shape[i] < reference_shape[i]:
                    pad_len = [(0, 0), (0, 0)]
                    delta = reference_shape[i] - output.shape[i]
                    pad_len[i] = (delta // 2, delta - delta // 2)
                    output = np.pad(output, pad_len, mode="constant")
                elif output.shape[i] > reference_shape[i]:
                    slicer = [slice(None)] * 2
                    delta = output.shape[i] - reference_shape[i]
                    slicer[i] = slice(delta // 2, delta // 2 + reference_shape[i])
                    output = output[tuple(slicer)]
            else:
                raise ValueError(
                    f"Invalid value for image_coordinates_origin: {self.image_coordinates_origin}"
                )
        return output

    def prepare_image_for_feature_matching(self, image: np.ndarray) -> np.ndarray:
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image, dtype=np.float32)
        return image

    def adjust_points_for_origin(
        self,
        points: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        if self.image_coordinates_origin == "center":
            center = np.array([(image_shape[1] - 1) / 2, (image_shape[0] - 1) / 2])
            return points - center
        if self.image_coordinates_origin == "top_left":
            return points
        raise ValueError(
            f"Invalid value for image_coordinates_origin: {self.image_coordinates_origin}"
        )

    def feature_based_registration(
        self,
        image_t: np.ndarray,
        image_r: np.ndarray,
    ) -> np.ndarray:
        image_t_float = self.prepare_image_for_feature_matching(image_t)
        image_r_float = self.prepare_image_for_feature_matching(image_r)
        sift_t = feature.SIFT()
        sift_r = feature.SIFT()
        sift_t.detect_and_extract(image_t_float)
        sift_r.detect_and_extract(image_r_float)

        descriptors_t = sift_t.descriptors
        descriptors_r = sift_r.descriptors
        keypoints_t = sift_t.keypoints
        keypoints_r = sift_r.keypoints

        if (
            descriptors_t is None
            or descriptors_r is None
            or descriptors_t.size == 0
            or descriptors_r.size == 0
        ):
            raise RuntimeError("SIFT feature detection failed to find descriptors.")

        matches = feature.match_descriptors(
            descriptors_t,
            descriptors_r,
            metric="euclidean",
            cross_check=True,
            max_ratio=0.75,
        )

        if matches.shape[0] < 3:
            raise RuntimeError("Not enough SIFT matches to estimate translation.")

        pts_t = keypoints_t[matches[:, 0]][:, ::-1]
        pts_r = keypoints_r[matches[:, 1]][:, ::-1]
        pts_t = self.adjust_points_for_origin(pts_t, image_t.shape)
        pts_r = self.adjust_points_for_origin(pts_r, image_r.shape)
        deltas = pts_r - pts_t
        offset_x = np.median(deltas[:, 0])
        offset_y = np.median(deltas[:, 1])
        return np.array([offset_y, offset_x])
