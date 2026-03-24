from io import BytesIO
import base64
import datetime
import io
import re
from typing import List, Literal, Tuple

import numpy as np
from PIL import Image


def get_timestamp(as_int: bool = False) -> str | int:
    """Return the current timestamp."""
    if as_int:
        return int(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def numpy_to_base64_image(arr: np.ndarray, format: str = "PNG") -> str:
    """Convert a NumPy array to a base64-encoded image string."""
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
    """Encode an image object or image file to base64."""
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
    """Decode a base64-encoded image."""
    image = Image.open(BytesIO(base64.b64decode(base64_data)))
    if return_type == "numpy":
        return np.asarray(image)
    if return_type == "pil":
        return image
    raise ValueError(f"Invalid return type: {return_type}")


def get_image_paths_from_text(
    text: str,
    return_text_without_image_tag: bool = False,
) -> List[str] | Tuple[List[str], str]:
    """Extract image paths from `<img ...>` tags."""
    paths = re.findall(r"<img (.*?)>", text)
    if return_text_without_image_tag:
        return paths, re.sub(r"<img .*?>", "", text)
    return paths


def get_image_path_from_text(text: str) -> str | None:
    """Extract the first image path from text containing `<img ...>` tags."""
    paths = get_image_paths_from_text(text)
    return paths[0] if len(paths) > 0 else None
