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

