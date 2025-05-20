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
    