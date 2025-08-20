from typing import Annotated, Dict, List, Any, Literal
import logging

import numpy as np
import scipy.ndimage as ndi

from eaa.tools.base import BaseTool, check, ToolReturnType
from eaa.tools.imaging.acquisition import AcquireImage
from eaa.image_proc import windowed_phase_cross_correlation

logger = logging.getLogger(__name__)


class ImageRegistration(BaseTool):
    """
    A tool that registers the latest image collected by the
    image acquisition tool and a reference image.
    """

    name: str = "image_registration"

    @check
    def __init__(
        self,
        image_acquisition_tool: AcquireImage,
        reference_image: np.ndarray = None,
        reference_pixel_size: float = 1.0,
        image_coordinates_origin: Literal["top_left", "center"] = "top_left",
        *args,
        **kwargs,
    ):
        """
        Initialize the image registration tool.

        Parameters
        ----------
        image_acquisition_tool : AcquireImage
            The image acquisition tool object. Do not use a copy of the object;
            pass the exact object that is used by the task manager so that the
            registration tool can access the latest acquired image.
        reference_image : np.ndarray, optional
            The reference image to register the latest image with.
        reference_pixel_size : float, optional
            The pixel size of the reference image.
        image_coordinates_origin : Literal["top_left", "center"], optional
            The origin of the image coordinates. Useful to handle cases where
            the size of the registered images do not match. Image registration
            finds the offset of the test image from the version aligned with the
            reference image where the values are the same for the same coordinates
            (i, j). When the origin defining the coordinates is different, the
            test image is cropped/padded differently when its size does not match.
            When this argument is set to "center", the test image is padded/cropped
            centrally. When it is set to "top_left", the test image is on the bottom
            and right sides.
        """
        super().__init__(*args, **kwargs)

        self.image_acquisition_tool = image_acquisition_tool
        self.reference_image = reference_image
        self.reference_pixel_size = reference_pixel_size
        self.image_coordinates_origin = image_coordinates_origin

        self.exposed_tools: List[Dict[str, Any]] = [
            {
                "name": "get_offset_of_latest_image",
                "function": self.get_offset_of_latest_image,
                "return_type": ToolReturnType.LIST,
            }
        ]

    def set_reference_image(
        self, reference_image: np.ndarray, 
        reference_pixel_size: float = 1.0
    ):
        """
        Set the reference image.
        """
        self.reference_image = reference_image
        self.reference_pixel_size = reference_pixel_size
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image to prepare it for registration.
        If the input image is a 3D array, the last dimension is assumed to
        be the channel dimension and will be averaged over.
        
        Parameters
        ----------
        image : np.ndarray
            A 2D or 3D array representing an image.

        Returns
        -------
        image : np.ndarray
            A 2D array representing an image.
        """
        if image.ndim == 3:
            image = np.mean(image, axis=-1)
        return image

    def get_offset_of_latest_image(
        self,
        register_with: Annotated[
            Literal["previous", "first", "reference"],
            "The image to register the latest image with. "
            "Can be 'previous', 'first', or 'reference'. "
            "'previous': register with the image collected by the acquisition tool before the latest. "
            "'first': register with the first image collected by the acquisition tool. "
            "'reference': register with the reference image provided to the tool. ",
        ],
    ) -> Annotated[
        List[float],
        "The translational offset [dy (vertical), dx (horizontal)] of the latest acquired "
        "image compared to the reference image. If the image is shifted to the right compared "
        "to the reference, the result will have a positive x-component; if the image is shifted "
        "to the bottom, the result will have a positive y-component. The returned values are in "
        "physical units, i.e., pixel size is already accounted for.",
    ]:
        """
        Register the latest image collected by the image acquisition tool
        and the reference image.
        """
        # Get the latest image:
        image_t = self.process_image(self.image_acquisition_tool.image_k)
        psize_t = self.image_acquisition_tool.psize_k

        if register_with == "previous":
            image_r = self.process_image(self.image_acquisition_tool.image_km1)
            psize_r = self.image_acquisition_tool.psize_km1
        elif register_with == "first":
            image_r = self.process_image(self.image_acquisition_tool.image_0)
            psize_r = self.image_acquisition_tool.psize_0
        elif register_with == "reference":
            if self.reference_image is None:
                raise ValueError("Reference image is not set.")
            image_r = self.process_image(self.reference_image)
            psize_r = self.reference_pixel_size
        else:
            raise ValueError(f"Invalid value for register_with: {register_with}")
        
        offset = self.register_images(image_t, image_r, psize_t, psize_r)
        return offset

    def register_images(
        self, 
        image_t: np.ndarray, 
        image_r: np.ndarray, 
        psize_t: float, 
        psize_r: float
    ) -> np.ndarray:
        """
        Register the target image with the reference image.
        
        Parameters
        ----------
        image_t : np.ndarray
            The target image.
        image_r : np.ndarray
            The reference image.
        psize_t : float
            The pixel size of the target image.
        psize_r : float
            The pixel size of the reference image.

        Returns
        -------
        np.ndarray
            The offset of the target image with respect to the reference image. If the
            target image is shifted to the right compared to the reference image, the
            result will have a positive x-component; if the target image is shifted
            to the bottom, the result will have a positive y-component. The returned
            values are in physical units, i.e., pixel size is already accounted for.
        """
        # Handle pixel size and image size differences
        if psize_t != psize_r:
            # Resize the target image to have the same pixel size as the reference image
            image_t = ndi.zoom(image_t, psize_t / psize_r)
        
        # Crop or pad the test image to match reference image size
        if image_t.shape != image_r.shape:
            for i in range(2):
                if self.image_coordinates_origin == "top_left":
                    if image_t.shape[i] < image_r.shape[i]:
                        pad_len = [(0, 0), (0, 0)]
                        pad_len[i] = (0, image_r.shape[i] - image_t.shape[i])
                        image_t = np.pad(image_t, pad_len, mode="constant")
                    elif image_t.shape[i] > image_r.shape[i]:
                        slicer = [slice(None)] * 2
                        slicer[i] = slice(0, image_r.shape[i])
                        image_t = image_t[tuple(slicer)]
                elif self.image_coordinates_origin == "center":
                    if image_t.shape[i] < image_r.shape[i]:
                        pad_len = [(0, 0), (0, 0)]
                        d = image_r.shape[i] - image_t.shape[i]
                        pad_len[i] = (d // 2, d - d // 2)
                        image_t = np.pad(image_t, pad_len, mode="constant")
                    elif image_t.shape[i] > image_r.shape[i]:
                        slicer = [slice(None)] * 2
                        d = image_t.shape[i] - image_r.shape[i]
                        slicer[i] = slice(d // 2, d // 2 + image_r.shape[i])
                        image_t = image_t[tuple(slicer)]
                else:
                    raise ValueError(
                        f"Invalid value for image_coordinates_origin: {self.image_coordinates_origin}"
                    )
        
        offset = windowed_phase_cross_correlation(image_t, image_r)
        
        # Convert the offset from pixel units to physical units. We use psize_r here
        # since the target image has already been resized to have the same pixel size
        # as the reference image.
        if psize_t != psize_r:
            offset = offset * psize_r
        return offset
