import numpy as np


def stitch_images(images: list[np.ndarray], gap: int = 0) -> np.ndarray:
    """Stitch a list of images together.
    
    Parameters
    ----------
    images : list[np.ndarray]
        A list of images to stitch together.
        
    Returns
    -------
    stitched_image : np.ndarray
        The stitched image.
    """
    max_shape = (
        max([img.shape[0] for img in images]),
        np.sum([img.shape[1] for img in images]) + gap * (len(images) - 1),
    )
    buffer = np.zeros(max_shape, dtype=images[0].dtype)
    
    x = 0
    for img in images:
        buffer[:img.shape[0], x : x + img.shape[1]] = img
        x += img.shape[1] + gap
    return buffer


def windowed_phase_cross_correlation(
    moving: np.ndarray, 
    ref: np.ndarray, 
) -> np.ndarray:
    """Phase correlation with windowing.

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
