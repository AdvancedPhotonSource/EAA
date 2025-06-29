from typing import Tuple, Literal, Optional
import datetime
import base64
import io
import re
from io import BytesIO
import os
import time
import logging
from math import inf

from PIL import Image
import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def to_tensor(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        # If CUDA is available, convert array to tensor using torch.tensor, which honors the set default device.
        # Otherwise, use from_numpy which creates a reference to the data in memory instead of creating a copy.
        if torch.cuda.is_available():
            return torch.tensor(x)
        else:
            try:
                return torch.from_numpy(x)
            except TypeError:
                return torch.tensor(x)
    else:
        return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return np.asarray(x)
    else:
        return x
    
    
def numpy_to_base64_image(arr: np.ndarray, format: str = "PNG") -> str:
    """
    Convert a NumPy array to base64-encoded image data URL.
    """
    # Normalize and convert to uint8 if needed
    if arr.dtype != np.uint8:
        arr = ((arr - arr.min()) / (arr.ptp() + 1e-5) * 255).astype(np.uint8)

    if arr.ndim == 2:  # grayscale
        mode = "L"
    elif arr.ndim == 3 and arr.shape[2] == 3:  # RGB
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA
        mode = "RGBA"
    else:
        raise ValueError("Unsupported array shape for image encoding")

    # Convert to PIL image
    image = Image.fromarray(arr, mode=mode)

    # Encode as base64
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return base64_str


def encode_image_base64(
    image: np.ndarray = None, 
    image_path: str = None
) -> str:
    """Encode an image to a base64 string.
    
    Parameters
    ----------
    image : np.ndarray, optional
        The image to encode. Exclusive with `image_path`.
    image_path : str, optional
        The path to the image to encode. Exclusive with `image`.
        
    Returns
    -------
    str
    """
    if image is not None and image_path is not None:
        raise ValueError("Only one of `image` or `image_path` should be provided.")
    
    if image_path is not None:
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
    elif image is not None:
        base64_data = numpy_to_base64_image(image)
    else:
        raise ValueError("Either `image` or `image_path` should be provided.")

    return base64_data


def decode_image_base64(
    base64_data: str, 
    return_type: Literal["numpy", "pil"] = "numpy"
) -> np.ndarray | Image.Image:
    """Decode a base64-encoded image to a NumPy array or PIL image.
    
    Parameters
    ----------
    base64_data : str
        The base64-encoded image data.
    return_type : Literal["numpy", "pil"], optional
        The type of the returned image.
        
    Returns
    -------
    np.ndarray | Image.Image
        The decoded image.
    """
    pil_image = Image.open(BytesIO(base64.b64decode(base64_data)))
    if return_type == "numpy":
        return np.array(pil_image)
    elif return_type == "pil":
        return pil_image
    else:
        raise ValueError(f"Invalid return type: {return_type}")


def get_image_path_from_text(
    text: str, 
    return_text_without_image_tag: bool = False
) -> str | None | Tuple[str, str]:
    """Get the path of an image from a string. The image path
    is assumed to be in the format of
    
    ```
    <img path/to/image.png>
    ```

    The path must be without any additional spaces or newlines in it.
    If no path is found, the function will return `None`.
    
    Parameters
    ----------
    text : str
        The text to get the image path from.
    return_text_without_image_tag : bool, optional
        If True, the function will also return the original text with the
        image tag of `<img path/to/image.png>` removed.
        
    Returns
    -------
    str | None
        The path to the image.
    """
    res = re.search(r'<img (.*?)>', text)
    if res:
        image_tag = res.group(0)
        path = res.group(1)
        if return_text_without_image_tag:
            text = text.replace(image_tag, "")
    else:
        path = None
    if return_text_without_image_tag:
        return path, text
    else:
        return path


def wait_for_file(
    file_path: str,
    duration: int = 30,
    timeout: Optional[int] = inf
) -> bool:
    """Wait for a file to exist and stay unchanged for a given timeout.
    The function only returns when the given file 
    (1) exists, and
    (2) its modification time does not change for `duration` seconds.
    
    If `timeout` is given, the function returns False if the above conditions
    are not met within `timeout` seconds.
    
    Parameters
    ----------
    file_path : str
        The path to the file to wait for.
    duration : int, optional
        The duration that the file is expected to stay unchanged to be
        considered as fully written.
    timeout : int, optional
        The timeout to wait for the file to exist and stay unchanged,
        given in seconds.
        
    Returns
    -------
    bool
        True if the file exists and stays unchanged for given a duration 
        within the given timeout, 
        False otherwise.
    """
    time_diff = 0
    time_mod = 0
    while any([time_diff < duration, not os.path.exists(file_path)]):
        if timeout is not None:
            if time_diff > timeout:
                return False
        time.sleep(1)
        if os.path.exists(file_path):
            if os.path.getmtime(file_path) != time_mod:
                time_mod = os.path.getmtime(file_path)
            time_diff = time.time() - time_mod
            logger.info(f"File {file_path} exists.")
            logger.info(f"Watching file and wait until the file doesn't change for {duration} seconds to process.")
        else:
            logger.info(f"File {file_path} does not exist.")
            logger.info(f"Waiting for {duration} seconds to process.")
            time.sleep(duration)
    return True
