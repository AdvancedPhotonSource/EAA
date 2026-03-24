import io
import logging
from typing import Literal

import numpy as np
import requests
import tifffile
from eaa.core.tooling.base import BaseTool, check, tool

from eaa.tool.imaging.acquisition import AcquireImage

logger = logging.getLogger(__name__)


class NNRegistration(BaseTool):
    """Tool that queries a NN server to obtain a registration offset between
    a reference image and the current image.

    The server must serve a model with ``prediction_type="offset"``, which
    accepts ``ref_image`` and ``test_image`` and returns
    ``{"offset_y": float, "offset_x": float}`` as fractions of the reference
    image size.
    The offset convention matches :class:`~eaa.tool.imaging.registration.ImageRegistration`:
    the returned values are the translation to apply to the test image so it
    aligns with the reference.
    """

    name: str = "nn_registration"

    @check
    def __init__(
        self,
        server_url: str,
        image_acquisition_tool: AcquireImage,
        require_approval: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the NN registration tool.

        Parameters
        ----------
        server_url : str
            Base URL of the inference server, e.g. ``"http://localhost:8090"``.
        image_acquisition_tool : AcquireImage
            The image acquisition tool instance.  Must be the same object used
            by the task manager so that its image buffers and call history
            reflect the current state.
        """
        super().__init__(*args, require_approval=require_approval, **kwargs)
        self.server_url = server_url.rstrip("/")
        self.image_acquisition_tool = image_acquisition_tool

    @staticmethod
    def _encode_as_tiff(image: np.ndarray) -> bytes:
        buf = io.BytesIO()
        tifffile.imwrite(buf, image.astype(np.float32))
        return buf.getvalue()

    @tool(name="get_offset")
    def get_offset(self, target: Literal["previous", "initial"] = "initial") -> np.ndarray:
        """Query the server and return the registration offset in physical units.

        Parameters
        ----------
        target : Literal["previous", "initial"]
            The reference image to register against.
            "previous": register the current image (``image_k``) against the
            immediately preceding image (``image_km1``).
            "initial": register the current image (``image_k``) against the
            first acquired image (``image_0``), giving the cumulative drift from
            the initial acquisition.

        Returns
        -------
        np.ndarray
            ``[offset_y, offset_x]`` in physical coordinate units (same units
            as the image acquisition positions).  The convention matches
            :meth:`~eaa.tool.imaging.registration.ImageRegistration.register_images`:
            this is the translation to apply to the current image so it aligns
            with the reference image.

        Raises
        ------
        RuntimeError
            If the required image buffers or acquisition history are not populated.
        requests.HTTPError
            If the server returns a non-2xx response.
        """
        acq = self.image_acquisition_tool

        if acq.image_k is None:
            raise RuntimeError(
                "Current image buffer (image_k) is not populated. "
                "Acquire at least one image before calling get_offset."
            )
        if not acq.image_acquisition_call_history:
            raise RuntimeError("No image acquisition history found.")

        if target == "previous":
            if acq.image_km1 is None:
                raise RuntimeError(
                    "Previous image buffer (image_km1) is not populated. "
                    "Acquire at least two images before calling get_offset with target='previous'."
                )
            ref_image = acq.image_km1
            ref_img_info = acq.image_acquisition_call_history[-2]
        elif target == "initial":
            if acq.image_0 is None:
                raise RuntimeError(
                    "Initial image buffer (image_0) is not populated. "
                    "Acquire at least one image before calling get_offset with target='initial'."
                )
            ref_image = acq.image_0
            ref_img_info = acq.image_acquisition_call_history[0]
        else:
            raise ValueError(f"`target` must be 'previous' or 'initial', got {target!r}.")

        ref_tiff = self._encode_as_tiff(ref_image)
        test_tiff = self._encode_as_tiff(acq.image_k)

        response = requests.post(
            f"{self.server_url}/predict",
            files={
                "ref_image": ("ref_image.tif", ref_tiff, "image/tiff"),
                "test_image": ("test_image.tif", test_tiff, "image/tiff"),
            },
        )
        response.raise_for_status()
        result = response.json()

        # Convert fractions of the reference image size to physical units.
        offset_y_phys = float(result["offset_y"]) * float(ref_img_info["size_y"])
        offset_x_phys = float(result["offset_x"]) * float(ref_img_info["size_x"])

        logger.debug(
            "NNRegistration offset (target=%s): frac=(%.4f, %.4f), phys=(%.4f, %.4f)",
            target, result["offset_y"], result["offset_x"],
            offset_y_phys, offset_x_phys,
        )
        return np.array([offset_y_phys, offset_x_phys], dtype=float)
