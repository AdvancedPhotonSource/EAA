from typing import Literal, Optional, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def stitch_images(
    images: list[np.ndarray | Image.Image], 
    gap: int = 0
) -> np.ndarray | Image.Image:
    """Stitch a list of images together.
    
    Parameters
    ----------
    images : list[np.ndarray | Image.Image]
        A list of images to stitch together.
    gap : int, optional
        The horizontal gap between the images.
        
    Returns
    -------
    stitched_image : np.ndarray | Image.Image
        The stitched image.
    """
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


def windowed_phase_cross_correlation(
    moving: np.ndarray, 
    ref: np.ndarray, 
) -> np.ndarray:
    """Phase correlation with windowing. The result gives
    the offset of the moving image with respect to the reference image.
    If the moving image is shifted to the right, the result will have a
    positive x-component; if the moving image is shifted to the bottom,
    the result will have a positive y-component.

    Parameters
    ----------
    moving : np.ndarray
        A 2D image.
    ref : np.ndarray
        A 2D image.

    Returns
    -------
    np.ndarray
        The shift of the moving image with respect to the reference image.
    """
    assert np.all(np.array(moving.shape) == np.array(ref.shape)), (
        "The shapes of the moving and reference images must be the same."
    )
    win_y = np.hanning(moving.shape[0])
    win_x = np.hanning(moving.shape[1])
    win = np.outer(win_y, win_x)
    
    f_moving = np.fft.fft2(moving * win)
    f_ref = np.fft.fft2(ref * win)
    
    f_corr = f_moving * f_ref.conj()
    f_corr = f_corr / np.abs(f_corr)
    
    map = np.fft.ifft2(f_corr).real
    shift = np.array(np.unravel_index(np.argmax(map), map.shape))
    for i in range(2):
        if shift[i] > map.shape[i] / 2:
            shift[i] -= map.shape[i]
    return shift


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
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image, cmap='gray')
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
                color="gray"
            )
            ax.hlines(
                marker_params["y"], 
                xmin=ax.get_xlim()[0], 
                xmax=ax.get_xlim()[1], 
                linestyle="--",
                linewidth=0.5,
                color="gray"
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
