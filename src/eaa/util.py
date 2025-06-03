import datetime
import base64
import io

from PIL import Image
import numpy as np
import torch


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
    