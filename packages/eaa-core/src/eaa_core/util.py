import base64
import datetime
import io
import logging
import os
import re
import time
from io import BytesIO
from math import inf
from typing import Any, Literal, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def to_tensor(x: Any) -> Any:
    """Convert array-like input to a torch tensor when appropriate.

    Parameters
    ----------
    x : Any
        Input value to convert.

    Returns
    -------
    Any
        Tensor-converted value for array-like inputs, otherwise the input.
    """
    if isinstance(x, (np.ndarray, list, tuple)):
        if torch.cuda.is_available():
            return torch.tensor(x)
        try:
            return torch.from_numpy(x)
        except TypeError:
            return torch.tensor(x)
    return x


def to_numpy(x: Any) -> Any:
    """Convert tensor or sequence input to a NumPy array when appropriate.

    Parameters
    ----------
    x : Any
        Input value to convert.

    Returns
    -------
    Any
        NumPy-converted value for supported inputs, otherwise the input.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return x


def wait_for_file(
    file_path: str,
    duration: int = 30,
    timeout: Optional[int] = inf,
) -> bool:
    """Wait for a file to exist and stop changing.

    Parameters
    ----------
    file_path : str
        Path to the file to watch.
    duration : int, optional
        Stable period in seconds required before returning success.
    timeout : int, optional
        Maximum wait time in seconds. ``None`` disables the timeout.

    Returns
    -------
    bool
        ``True`` when the file exists and remains unchanged for ``duration``
        seconds within the timeout window, otherwise ``False``.
    """
    time_diff = 0.0
    time_mod = 0.0
    while any([time_diff < duration, not os.path.exists(file_path)]):
        if timeout is not None and time_diff > timeout:
            return False
        time.sleep(1)
        if os.path.exists(file_path):
            if os.path.getmtime(file_path) != time_mod:
                time_mod = os.path.getmtime(file_path)
            time_diff = time.time() - time_mod
            logger.info("File %s exists.", file_path)
            logger.info(
                "Watching file and wait until the file doesn't change for %s seconds to process.",
                duration,
            )
        else:
            logger.info("File %s does not exist.", file_path)
            logger.info("Waiting for %s seconds to process.", duration)
            time.sleep(duration)
    return True


def get_timestamp(as_int: bool = False) -> str | int:
    """Return the current timestamp.

    Parameters
    ----------
    as_int : bool, optional
        When ``True``, return the timestamp as an integer.

    Returns
    -------
    str | int
        Formatted timestamp.
    """
    if as_int:
        return int(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def numpy_to_base64_image(arr: np.ndarray, format: str = "PNG") -> str:
    """Convert a NumPy array to a base64-encoded image string.

    Parameters
    ----------
    arr : numpy.ndarray
        Image array to encode.
    format : str, optional
        Image format used for serialization.

    Returns
    -------
    str
        Base64-encoded image data.
    """
    if arr.dtype != np.uint8:
        arr = ((arr - arr.min()) / (np.ptp(arr) + 1e-5) * 255).astype(np.uint8)

    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError("Unsupported array shape for image encoding.")

    image = Image.fromarray(arr, mode=mode)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_image_base64(
    image: np.ndarray | Image.Image | None = None,
    image_path: str | None = None,
) -> str:
    """Encode an image object or image file to base64.

    Parameters
    ----------
    image : numpy.ndarray | PIL.Image.Image | None, optional
        In-memory image to encode.
    image_path : str | None, optional
        Path to an image file to encode.

    Returns
    -------
    str
        Base64-encoded image data.
    """
    if image is not None and image_path is not None:
        raise ValueError("Only one of `image` or `image_path` should be provided.")
    if image_path is not None:
        with open(image_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")
    if image is None:
        raise ValueError("Either `image` or `image_path` should be provided.")
    if isinstance(image, np.ndarray):
        return numpy_to_base64_image(image)
    if isinstance(image, Image.Image):
        return numpy_to_base64_image(np.asarray(image))
    raise ValueError("Invalid image type. Must be a NumPy array or PIL image.")


def decode_image_base64(
    base64_data: str,
    return_type: Literal["numpy", "pil"] = "numpy",
) -> np.ndarray | Image.Image:
    """Decode a base64-encoded image.

    Parameters
    ----------
    base64_data : str
        Base64-encoded image content.
    return_type : {"numpy", "pil"}, optional
        Output representation.

    Returns
    -------
    numpy.ndarray | PIL.Image.Image
        Decoded image.
    """
    image = Image.open(BytesIO(base64.b64decode(base64_data)))
    if return_type == "numpy":
        return np.asarray(image)
    if return_type == "pil":
        return image
    raise ValueError(f"Invalid return type: {return_type}")


def get_image_paths_from_text(
    text: str,
    return_text_without_image_tag: bool = False,
) -> list[str] | tuple[list[str], str]:
    """Extract image paths from ``<img ...>`` tags in text.

    Parameters
    ----------
    text : str
        Input text that may contain image tags.
    return_text_without_image_tag : bool, optional
        When ``True``, also return the text with image tags removed.

    Returns
    -------
    list[str] | tuple[list[str], str]
        Extracted paths, optionally paired with cleaned text.
    """
    paths = re.findall(r"<img (.*?)>", text)
    if return_text_without_image_tag:
        return paths, re.sub(r"<img .*?>", "", text)
    return paths


def get_image_path_from_text(text: str) -> str | None:
    """Extract the first image path from text containing ``<img ...>`` tags.

    Parameters
    ----------
    text : str
        Input text that may contain image tags.

    Returns
    -------
    str | None
        First extracted image path, if present.
    """
    paths = get_image_paths_from_text(text)
    return paths[0] if len(paths) > 0 else None
