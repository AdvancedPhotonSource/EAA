from typing import Literal, Optional, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy import optimize
from scipy.special import erf
from skimage.metrics import normalized_mutual_information
from skimage.registration import phase_cross_correlation as skimage_phase_cross_correlation
from sciagent.message_proc import generate_openai_message
from sciagent.task_manager.base import BaseTaskManager


def stitch_images(
    images: list[np.ndarray | Image.Image], 
    gap: int = 0,
    frame: bool = False
) -> np.ndarray | Image.Image:
    """Stitch a list of images together.
    
    Parameters
    ----------
    images : list[np.ndarray | Image.Image]
        A list of images to stitch together.
    gap : int, optional
        The horizontal gap between the images.
    frame : bool, optional
        If True, a frame is added to each image.
    Returns
    -------
    stitched_image : np.ndarray | Image.Image
        The stitched image.
    """
    def max_value_for_dtype(img: np.ndarray) -> float:
        dtype = img.dtype
        if np.issubdtype(dtype, np.integer):
            return float(np.iinfo(dtype).max)
        else:
            return img.max()

    def should_use_white_frame(image_array: np.ndarray) -> bool:
        max_value = max_value_for_dtype(image_array)
        threshold = 128 if max_value >= 255 else max_value / 2
        return np.mean(image_array < threshold) > 0.5

    def add_frame_to_numpy(image_array: np.ndarray) -> np.ndarray:
        frame_width = 1
        framed = image_array.copy()
        if image_array.ndim == 2 or image_array.shape[2] == 1:
            max_value = max_value_for_dtype(image_array)
            frame_value = max_value if should_use_white_frame(image_array) else 0
            framed[:frame_width, :] = frame_value
            framed[-frame_width:, :] = frame_value
            framed[:, :frame_width] = frame_value
            framed[:, -frame_width:] = frame_value
            return framed

        max_value = max_value_for_dtype(image_array)
        frame_color = np.zeros(image_array.shape[2], dtype=image_array.dtype)
        frame_color[0] = max_value
        if frame_color.shape[0] > 3:
            frame_color[3] = max_value
        framed[:frame_width, :, :] = frame_color
        framed[-frame_width:, :, :] = frame_color
        framed[:, :frame_width, :] = frame_color
        framed[:, -frame_width:, :] = frame_color
        return framed

    def add_frame_to_pil(image: Image.Image) -> Image.Image:
        frame_width = 1
        image_copy = image.copy()
        if image.mode in {"RGB", "RGBA"}:
            frame_color = (255, 0, 0, 255) if image.mode == "RGBA" else (255, 0, 0)
        else:
            image_array = np.array(image)
            max_value = max_value_for_dtype(image_array)
            frame_color = int(max_value) if should_use_white_frame(image_array) else 0

        draw = ImageDraw.Draw(image_copy)
        draw.rectangle(
            [0, 0, image_copy.width - 1, image_copy.height - 1],
            outline=frame_color,
            width=frame_width,
        )
        return image_copy

    if frame:
        print("Adding frame to images")
        images = [
            add_frame_to_numpy(img) if isinstance(img, np.ndarray) else add_frame_to_pil(img)
            for img in images
        ]
    if isinstance(images[0], np.ndarray):
        max_shape = (
            max([img.shape[0] for img in images]),
            np.sum([img.shape[1] for img in images]) + gap * (len(images) - 1),
        )
        buffer = np.zeros(max_shape, dtype=images[0].dtype)
        
        x = 0
        for img in images:
            buffer[:img.shape[0], x : x + img.shape[1]] = img
            x += img.shape[1] + gap
    elif isinstance(images[0], Image.Image):
        max_shape = (
            sum([img.width for img in images]) + gap * (len(images) - 1),
            max([img.height for img in images]),
        )
        buffer = Image.new("RGB", max_shape)
        x = 0
        for img in images:
            buffer.paste(img, (x, 0))
            x += img.width + gap
    else:
        raise ValueError("The images must be either numpy arrays or PIL images.")
        
    return buffer


def _gaussian_rect_window(shape: tuple[int, int], decay_fraction: float = 0.2) -> np.ndarray:
    """2D Gaussian-softened rectangle window.

    The mask is 1 in the center and decays smoothly to 0 at each edge.
    The decay is the convolution of a step function with a Gaussian, implemented
    via the error function. The transition from 1 to 0 spans ``decay_fraction``
    of the image size on each side.
    """
    def _win_1d(size: int) -> np.ndarray:
        decay_len = decay_fraction * size
        sigma = decay_len / 4.0
        s2 = sigma * np.sqrt(2.0)
        x = np.arange(size, dtype=float)
        left = 0.5 * (1.0 + erf((x - decay_len / 2.0) / s2))
        right = 0.5 * (1.0 - erf((x - (size - 1.0 - decay_len / 2.0)) / s2))
        return np.minimum(left, right)

    return np.outer(_win_1d(shape[0]), _win_1d(shape[1]))


def phase_cross_correlation(
    moving: np.ndarray,
    ref: np.ndarray,
    filtering_method: Optional[Literal["hanning", "gaussian"]] = "hanning",
    upsample_factor: int = 1,
) -> np.ndarray | Tuple[np.ndarray, float]:
    """Phase correlation with windowing. The result gives
    the translation offset to apply to the moving image so that it aligns
    with the reference image.

    Parameters
    ----------
    moving : np.ndarray
        A 2D image.
    ref : np.ndarray
        A 2D image.
    filtering_method : {"hanning", "gaussian"} or None, optional
        Window function applied to both images before phase correlation to
        reduce spectral leakage.  ``"hanning"`` uses a standard Hanning window.
        ``"gaussian"`` uses a Gaussian-softened rectangle that is 1 in the
        centre and decays to 0 at each edge over 20 % of the image size.
        Pass ``None`` to disable windowing.
    upsample_factor : int, optional
        Upsampling factor for subpixel accuracy in phase correlation.
        A value of 1 yields pixel-level precision.

    Returns
    -------
    np.ndarray
        The translation offset (dy, dx) to apply to the moving image.
    """
    assert np.all(np.array(moving.shape) == np.array(ref.shape)), (
        "The shapes of the moving and reference images must be the same."
    )
    moving = moving - moving.mean()
    ref = ref - ref.mean()
    if filtering_method == "hanning":
        win_y = np.hanning(moving.shape[0])
        win_x = np.hanning(moving.shape[1])
        win = np.outer(win_y, win_x)
        moving_for_registration = moving * win
        ref_for_registration = ref * win
    elif filtering_method == "gaussian":
        win = _gaussian_rect_window(moving.shape, decay_fraction=0.2)
        moving_for_registration = moving * win
        ref_for_registration = ref * win
    elif filtering_method is None:
        moving_for_registration = moving
        ref_for_registration = ref
    else:
        raise ValueError(
            f"Unknown filtering_method {filtering_method!r}. "
            "Use 'hanning', 'gaussian', or None."
        )

    shift, _, _ = skimage_phase_cross_correlation(
        ref_for_registration,
        moving_for_registration,
        upsample_factor=upsample_factor,
    )
    return shift


def error_minimization_registration(
    moving: np.ndarray,
    ref: np.ndarray,
    y_valid_fraction: float = 0.8,
    x_valid_fraction: float = 0.8,
    subpixel: bool = True,
) -> np.ndarray:
    """Image registration by exhaustive integer-shift MSE search with quadratic
    subpixel refinement.

    A central window of size ``(y_valid_fraction * h, x_valid_fraction * w)``
    is fixed in the reference image.  The moving image is sampled at the same
    window position for every integer shift (dy, dx) within the margins
    ``[-max_dy, max_dy] × [-max_dx, max_dx]``, where the margins are the pixel
    gaps between the valid window and the image boundary.  No wrap-around pixels
    are ever included: the valid window is identical for all shifts.

    The resulting 2-D MSE map is fitted with a 2-D quadratic polynomial.  The
    analytic minimum of that polynomial is returned as the sub-pixel shift.

    Parameters
    ----------
    moving : np.ndarray
        2-D image to register.
    ref : np.ndarray
        2-D reference image with the same shape as *moving*.
    y_valid_fraction : float
        Fraction of the image height occupied by the comparison window.
        Values close to 1 leave little margin and therefore a small search range.
    x_valid_fraction : float
        Same as *y_valid_fraction* along the x (column) axis.
    subpixel : bool
        If True, perform subpixel refinement using a 2D quadratic fit.

    Returns
    -------
    np.ndarray
        Estimated (dy, dx) shift to apply to *moving* so that it aligns with
        *ref*.
    """
    assert moving.shape == ref.shape, (
        "The shapes of the moving and reference images must be the same."
    )
    h, w = ref.shape

    vh = int(round(y_valid_fraction * h))
    vw = int(round(x_valid_fraction * w))

    # Centre the valid window; margin on each side = max search range
    r0 = (h - vh) // 2
    c0 = (w - vw) // 2
    r1, c1 = r0 + vh, c0 + vw
    max_dy, max_dx = r0, c0

    if max_dy == 0 and max_dx == 0:
        return np.zeros(2)

    dy_vals = np.arange(-max_dy, max_dy + 1)
    dx_vals = np.arange(-max_dx, max_dx + 1)

    ref_crop = ref[r0:r1, c0:c1].astype(float)
    moving_f = moving.astype(float)

    # Exhaustive integer-shift MSE map
    error_map = np.empty((len(dy_vals), len(dx_vals)))
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            diff = moving_f[r0 + dy : r1 + dy, c0 + dx : c1 + dx] - ref_crop
            error_map[i, j] = np.mean(diff * diff)

    # Fit quadratic in a local neighbourhood around the integer minimum.
    # Neighbourhood half-width: 10% of image size / 2, at least 1 (→ 3×3 minimum).
    min_i, min_j = np.unravel_index(np.argmin(error_map), error_map.shape)
    if not subpixel:
        return -np.array([float(dy_vals[min_i]), float(dx_vals[min_j])])

    half_y = max(1, int(round(0.05 * h)))
    half_x = max(1, int(round(0.05 * w)))
    i_lo = max(0, min_i - half_y)
    i_hi = min(len(dy_vals) - 1, min_i + half_y)
    j_lo = max(0, min_j - half_x)
    j_hi = min(len(dx_vals) - 1, min_j + half_x)
    local_dy = dy_vals[i_lo : i_hi + 1]
    local_dx = dx_vals[j_lo : j_hi + 1]
    local_err = error_map[i_lo : i_hi + 1, j_lo : j_hi + 1]

    # The 2-D quadratic has 6 parameters; require ≥3 points in each dimension so
    # the design matrix is well-determined and the Hessian is not rank-deficient.
    if len(local_dy) >= 3 and len(local_dx) >= 3:
        # Fit: f(dy, dx) = a*dy² + b*dx² + c*dy*dx + d*dy + e*dx + g
        dy_mesh, dx_mesh = np.meshgrid(local_dy, local_dx, indexing="ij")
        dy_f = dy_mesh.ravel()
        dx_f = dx_mesh.ravel()
        design = np.column_stack(
            [dy_f**2, dx_f**2, dy_f * dx_f, dy_f, dx_f, np.ones(len(dy_f))]
        )
        coeffs, _, _, _ = np.linalg.lstsq(design, local_err.ravel(), rcond=None)
        a, b, c, d, e, _ = coeffs

        # Analytic minimum: solve Hessian @ [dy_min, dx_min]ᵀ = -gradient
        # Hessian = [[2a, c], [c, 2b]]; gradient at origin = [d, e]
        hess = np.array([[2.0 * a, c], [c, 2.0 * b]])
        try:
            if np.all(np.linalg.eigvalsh(hess) > 0):
                shift = np.linalg.solve(hess, np.array([-d, -e]))
            else:
                raise np.linalg.LinAlgError("Hessian not positive definite")
        except np.linalg.LinAlgError:
            shift = np.array([float(dy_vals[min_i]), float(dx_vals[min_j])])
    else:
        shift = np.array([float(dy_vals[min_i]), float(dx_vals[min_j])])

    # Negate: the MSE is minimised at the offset where moving[r0+dy:] matches
    # ref[r0:], but the caller wants the shift to apply to moving so that
    # roll(moving, shift) ≈ ref, which is the opposite direction.
    return -shift


def normalize_image_01(image: np.ndarray) -> np.ndarray:
    """Normalize image intensities to [0, 1]."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return np.zeros_like(image, dtype=np.float32)


def warp_translation(
    image: np.ndarray,
    shift: np.ndarray | tuple[float, float] | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp an image by translation and return warped image with valid mask.

    Parameters
    ----------
    image : np.ndarray
        A 2D image to warp.
    shift : np.ndarray | tuple[float, float] | list[float]
        Internal translation in (dy, dx) used by the optimizer.
    """
    dy, dx = map(float, shift)
    rows, cols = image.shape
    y, x = np.indices((rows, cols), dtype=np.float32)
    sample_y = y - dy
    sample_x = x - dx
    warped = ndi.map_coordinates(
        image,
        [sample_y, sample_x],
        order=1,
        mode="constant",
        cval=0.0,
    )
    valid = (
        (sample_y >= 0)
        & (sample_y <= rows - 1)
        & (sample_x >= 0)
        & (sample_x <= cols - 1)
    )
    return warped, valid


def translation_nmi_registration(
    moving: np.ndarray,
    ref: np.ndarray,
    pyramid_levels: tuple[int, ...] = (4, 2, 1),
    bins: int = 64,
    sample_frac: float = 0.2,
    smooth_sigmas: Optional[dict[int, float]] = None,
    optimizer: Literal["powell", "nelder-mead"] = "nelder-mead",
    max_iter: int = 60,
    tol: float = 1e-4,
) -> np.ndarray:
    """Estimate translation (dy, dx) to apply to moving image by maximizing NMI."""
    if moving.ndim != 2 or ref.ndim != 2:
        raise ValueError("`moving` and `ref` must both be 2D images.")
    if sample_frac <= 0 or sample_frac > 1:
        raise ValueError("`sample_frac` must be in (0, 1].")
    if smooth_sigmas is None:
        smooth_sigmas = {4: 1.5, 2: 1.0, 1: 0.0}

    shift_full = np.zeros(2, dtype=np.float64)

    for level in pyramid_levels:
        zoom_factor = 1.0 / float(level)
        moving_l = ndi.zoom(moving, zoom_factor, order=1)
        ref_l = ndi.zoom(ref, zoom_factor, order=1)

        sigma = float(smooth_sigmas.get(level, 0.0))
        if sigma > 0:
            moving_l = ndi.gaussian_filter(moving_l, sigma=sigma)
            ref_l = ndi.gaussian_filter(ref_l, sigma=sigma)

        moving_l = normalize_image_01(moving_l)
        ref_l = normalize_image_01(ref_l)

        shift_level_0 = shift_full / float(level)

        def objective(shift_level: np.ndarray) -> float:
            warped, valid_warp = warp_translation(moving_l, shift_level)
            overlap_indices = np.flatnonzero(valid_warp.ravel())
            if overlap_indices.size < 8:
                return 0.0

            n_samples = max(8, int(overlap_indices.size * sample_frac))
            if n_samples < overlap_indices.size:
                rng = np.random.default_rng(13)
                chosen = rng.choice(overlap_indices, size=n_samples, replace=False)
            else:
                chosen = overlap_indices

            ref_samples = ref_l.ravel()[chosen]
            warped_samples = warped.ravel()[chosen]
            nmi = normalized_mutual_information(ref_samples, warped_samples, bins=bins)
            if not np.isfinite(nmi):
                return 1e6
            return -float(nmi)

        if optimizer == "powell":
            result = optimize.minimize(
                objective,
                shift_level_0,
                method="Powell",
                options={
                    "xtol": tol,
                    "ftol": tol,
                    "maxiter": max_iter,
                    "direc": np.diag([2.0, 2.0]),
                },
            )
        elif optimizer == "nelder-mead":
            simplex = np.vstack(
                [
                    shift_level_0,
                    shift_level_0 - np.array([2.0, 0.0]),
                    shift_level_0 - np.array([0.0, 2.0]),
                ]
            )
            result = optimize.minimize(
                objective,
                shift_level_0,
                method="Nelder-Mead",
                options={
                    "xatol": tol,
                    "fatol": tol,
                    "maxiter": max_iter,
                    "initial_simplex": simplex,
                },
            )
        else:
            raise ValueError(
                f"Unsupported optimizer '{optimizer}'. Use 'powell' or 'nelder-mead'."
            )

        shift_opt = result.x if result.success else shift_level_0
        shift_full = np.array(shift_opt, dtype=np.float64) * float(level)

    return shift_full


def physical_pos_to_pixel(
    physical_x: float | np.ndarray,
    physical_range: tuple[float, float],
    size: int
) -> float:
    """Convert a physical x-coordinate to a pixel coordinate.
    """
    return np.round(
        (physical_x - physical_range[0]) / (physical_range[1] - physical_range[0]) * size
    )
    

def physical_length_to_pixel(
    physical_length: float | np.ndarray,
    physical_range: tuple[float, float],
    size: int
) -> float:
    """Convert a physical length to a pixel length.
    """
    return np.round(
        physical_length / (physical_range[1] - physical_range[0]) * size
    )


def plot_2d_image(
    image: np.ndarray, 
    add_axis_ticks: bool = False,
    x_coords: Optional[List[float]] = None,
    y_coords: Optional[List[float]] = None,
    n_ticks: int = 10,
    add_grid_lines: bool = False,
    invert_yaxis: bool = False,
    cmap: str = "inferno"
) -> plt.Figure:
    """Save an image to the temporary directory.

    Parameters
    ----------
    image : np.ndarray
        The image to save.
    add_axis_ticks : bool, optional
        If True, axis ticks are added to the image to indicate positions.
    x_coords : List[float], optional
        The x-coordinates to add to the image. Required when `add_axis_ticks` is True.
        The length of this list must be the same as the number of columns in the image.
    y_coords : List[float], optional
        The y-coordinates to add to the image. Required when `add_axis_ticks` is True.
        The length of this list must be the same as the number of rows in the image.
    n_ticks : int, optional
        The number of ticks in each axis..
    add_grid_lines : bool, optional
        If True, grid lines are added to the image.
    invert_yaxis : bool, optional
        If True, the y-axis is inverted.
    cmap: str, optional
        The colormap for plotting the image.
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image, cmap=cmap)
    if add_axis_ticks:
        if x_coords is None:
            x_coords = np.arange(image.shape[1])
        if y_coords is None:
            y_coords = np.arange(image.shape[0])
        ax.set_xticks(np.linspace(0, len(x_coords) - 1, n_ticks, dtype=int))
        ax.set_yticks(np.linspace(0, len(y_coords) - 1, n_ticks, dtype=int))
        ax.set_xticklabels([np.round(x_coords[i], 2) for i in ax.get_xticks()])
        ax.set_yticklabels([np.round(y_coords[i], 2) for i in ax.get_yticks()])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(add_grid_lines)
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    return fig


def add_marker_to_imgae(
    image: Optional[np.ndarray | Image.Image] = None,
    ax: Optional[plt.Figure] = None,
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    marker_type: Literal["line", "rectangle"] = None,
    marker_params: dict = None,
    add_reticles_for_key_points: bool = False,
    cmap: str = "inferno"
) -> plt.Figure:
    """Add a marker to an image and plot it.

    Parameters
    ----------
    image : np.ndarray | Image.Image, optional
        The image array or PIL object. If this, instead of `ax`, is provided,
        a new figure will be created and the marker will be plotted on it.
    ax : plt.Figure, optional
        This axis that already has the image plotted on it. This should not
        be given simultaneously with `image`.
    x_range, y_range : tuple[float, float], optional
        The x- and y- physical range of the image. If not given, all
        coordinates will be assumed to be pixels.
        The x- and y- physical coordinates of the image. If not given, all
        coordinates will be assumed to be pixels.
    marker_type : Literal["line", "rectangle"], optional
        The type of marker to add.
    marker_params : dict, optional
        The parameters of the marker. Depending on `marker_type`, this
        argument expects different inputs:
        - "line":
            - `x`: tuple[float, float]. The starting and ending x-coordinates of the line.
            - `y`: tuple[float, float]. The starting and ending y-coordinates of the line.
        - "rectangle":
            - `x`: float. The x-coordinate of the top-left corner of the rectangle.
            - `y`: float. The y-coordinate of the top-left corner of the rectangle.
            - `width`: float. The width of the rectangle.
            - `height`: float. The height of the rectangle.
            - `fill`: bool. Whether to fill the rectangle.
        - common to all:
            - `color`: str. The color of the marker.
            - `linewidth`: float. The width of the marker.
            - `linestyle`: str. The style of the marker.
            - `alpha`: float. The transparency of the marker.
    add_reticles_for_key_points : bool, optional
        If True, reticles that extend to the axes are added to key points of the marker to
        help identify the coordinates of the key points.

    Returns
    -------
    plt.Figure
        The figure object.
    """
    if (image is None and ax is None) or (image is not None and ax is not None):
        raise ValueError("Exactly one of `image` or `ax` must be provided.")

    if marker_type is None:
        raise ValueError("`marker_type` must be given.")

    if image is not None:
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        fig = plot_2d_image(image)
        ax = fig.axes[0]

    if marker_type == "line":
        if x_range is not None and y_range is not None:
            marker_params["x"] = physical_pos_to_pixel(np.array(marker_params["x"]), x_range, image.shape[1])
            marker_params["y"] = physical_pos_to_pixel(np.array(marker_params["y"]), y_range, image.shape[0])
        ax.plot(
            marker_params["x"], marker_params["y"], 
            **{k: v for k, v in marker_params.items() if k != "x" and k != "y"}
        )
        if add_reticles_for_key_points:
            ax.vlines(
                marker_params["x"], 
                ymin=ax.get_ylim()[0],
                ymax=ax.get_ylim()[1], 
                linestyle="--",
                linewidth=0.5,
                color="inferno"
            )
            ax.hlines(
                marker_params["y"], 
                xmin=ax.get_xlim()[0], 
                xmax=ax.get_xlim()[1], 
                linestyle="--",
                linewidth=0.5,
                color="inferno"
            )
    elif marker_type == "rectangle":
        if x_range is not None and y_range is not None:
            marker_params["x"] = physical_pos_to_pixel(np.array(marker_params["x"]), x_range, image.shape[1])
            marker_params["y"] = physical_pos_to_pixel(np.array(marker_params["y"]), y_range, image.shape[0])
            marker_params["width"] = physical_length_to_pixel(marker_params["width"], x_range, image.shape[1])
            marker_params["height"] = physical_length_to_pixel(marker_params["height"], y_range, image.shape[0])
        ax.add_patch(
            plt.Rectangle(
                xy=(marker_params["x"], marker_params["y"]),
                **{k: v for k, v in marker_params.items() if k != "x" and k != "y"}
            )
        )
    else:
        raise ValueError(f"Invalid marker type: {marker_type}")
        
    return ax.get_figure()


def check_feature_presence_llm(
    task_manager: Optional[BaseTaskManager],
    image: np.ndarray,
    reference_image: np.ndarray,
    n_votes: int = 1,
    positive_examples: Optional[List[Image.Image]] = None,
    negative_examples: Optional[List[Image.Image]] = None,
) -> bool:
    """Lets an LLM judge if the features in the reference image 
    are present in the current image.

    Parameters
    ----------
    task_manager : Optional[BaseTaskManager]
        The task manager that owns the agent used to query the LLM.
    image : np.ndarray
        The current image that should be checked.
    reference_image : np.ndarray
        The reference image containing the feature of interest.
    n_votes : int, optional
        The number of votes to collect from the LLM.
    positive_examples : Optional[List[Image.Image]], optional
        Example images where the feature is present.
    negative_examples : Optional[List[Image.Image]], optional
        Example images where the feature is absent.
    
    Returns
    -------
    bool
        Whether the feature is present in the current image.
    """
    stitched_image = stitch_images([reference_image, image], gap=10, frame=True)
    prompt_lines = [
        "Are the non-periodic features in the image on the left also present in the image on the right?",
        "- Features don't have to be exactly aligned, and one may be blurrier than another.",
        "- 'Periodic features' refers to repeating patterns like grids, repeating dots, etc. "
        "They should not be considered as features.",
    ]
    example_context = []
    if positive_examples:
        prompt_lines.append(
            "- Positive examples (feature present) are provided before this query."
        )
        for idx, example in enumerate(positive_examples, start=1):
            example_context.append(
                generate_openai_message(
                    role="system",
                    content=f"Positive example {idx} (feature present).",
                    image=example,
                )
            )
    if negative_examples:
        prompt_lines.append(
            "- Negative examples (feature absent) are provided before this query."
        )
        for idx, example in enumerate(negative_examples, start=1):
            example_context.append(
                generate_openai_message(
                    role="system",
                    content=f"Negative example {idx} (feature absent).",
                    image=example,
                )
            )
    prompt_lines.append("- Just answer with 'yes' or 'no'.")
    message = generate_openai_message(
        role="system",
        content="\n".join(prompt_lines),
        image=stitched_image
    )
    votes = []
    for _ in range(n_votes):
        while True:
            response, outgoing = task_manager.agent.receive(
                message,
                context=example_context or None,
                return_outgoing_message=True
            )
            if task_manager is not None:
                task_manager.update_message_history(outgoing, update_context=False, update_full_history=True)
                task_manager.update_message_history(response, update_context=False, update_full_history=True)
            if "yes" in response["content"].lower() or "no" in response["content"].lower():
                votes.append(True if "yes" in response["content"].lower() else False)
                break
    return np.mean(votes) >= 0.5
