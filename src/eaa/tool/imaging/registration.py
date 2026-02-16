from typing import Annotated, List, Literal, Optional, Tuple
import logging
import json

import numpy as np
import scipy.ndimage as ndi
from skimage import feature
from sciagent.tool.base import BaseTool, check, ToolReturnType, tool

from eaa.tool.imaging.acquisition import AcquireImage
from eaa.image_proc import phase_cross_correlation, translation_nmi_registration

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
        registration_method: Literal["phase_correlation", "sift", "mutual_information"] = "phase_correlation",
        require_approval: bool = False,
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
        registration_method : Literal["phase_correlation", "sift", "mutual_information"], optional
            The method used to estimate translational offsets. "phase_correlation"
            uses phase correlation, "sift" uses feature matching, and
            "mutual_information" uses pyramid-based normalized mutual information.
        """
        super().__init__(*args, require_approval=require_approval, **kwargs)

        self.image_acquisition_tool = image_acquisition_tool
        self.reference_image = reference_image
        self.reference_pixel_size = reference_pixel_size
        self.image_coordinates_origin = image_coordinates_origin
        self.registration_method = registration_method

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

    @tool(name="get_offset_of_latest_image", return_type=ToolReturnType.LIST)
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
        psize_r: float,
        return_correlation_value: bool = False,
        use_hanning_window: bool = True,
        registration_method: Optional[Literal["phase_correlation", "sift", "mutual_information"]] = None,
    ) -> np.ndarray | Tuple[np.ndarray, float] | str:
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
        return_correlation_value : bool, optional
            If True, the correlation value is returned along with the offset.
        use_hanning_window : bool, optional
            If True, a Hanning window is used to smooth the images before the
            correlation is computed.
        registration_method : Optional[Literal["phase_correlation", "sift", "mutual_information"]], optional
            Overrides the default registration method for this call. If "sift" is
            used and `return_correlation_value` is True, the correlation value is
            reported as NaN. When using SIFT, image sizes are reconciled by the
            coordinate origin while pixel sizes are still normalized.

        Returns
        -------
        np.ndarray | str
            If `return_correlation_value` is False, the offset of the target 
            image with respect to the reference image is returned. If the
            target image is shifted to the right compared to the reference image, the
            result will have a positive x-component; if the target image is shifted
            to the bottom, the result will have a positive y-component. The returned
            values are in physical units, i.e., pixel size is already accounted for.
            If `return_correlation_value` is True, a stringified JSON object with the
            keys "offset" and "correlation_value" is returned.
        """
        method = registration_method or self.registration_method

        # Handle pixel size and image size differences
        if psize_t != psize_r:
            # Resize the target image to have the same pixel size as the reference image
            image_t = ndi.zoom(image_t, psize_t / psize_r)
        
        if method in {"phase_correlation", "mutual_information"}:
            image_t = self.reconcile_image_shape(image_t, image_r.shape)

        if method == "phase_correlation":
            if return_correlation_value:
                offset, correlation_value = phase_cross_correlation(
                    image_t,
                    image_r,
                    return_correlation_value=return_correlation_value,
                    use_hanning_window=use_hanning_window,
                )
            else:
                offset = phase_cross_correlation(
                    image_t,
                    image_r,
                    return_correlation_value=return_correlation_value,
                    use_hanning_window=use_hanning_window,
                )
        elif method == "mutual_information":
            offset = translation_nmi_registration(
                moving=image_t,
                ref=image_r,
                pyramid_levels=(4, 2, 1),
                bins=64,
                sample_frac=0.2,
                optimizer="nelder-mead",
                max_iter=60,
                tol=1e-4,
            )
            correlation_value = float("nan")
        elif method == "sift":
            offset = self.feature_based_registration(image_t, image_r)
            correlation_value = float("nan")
        else:
            raise ValueError(f"Invalid registration method: {method}")
        
        # Convert the offset from pixel units to physical units. We use psize_r here
        # since the target image has already been resized to have the same pixel size
        # as the reference image.
        offset = offset * psize_r
        if return_correlation_value:
            return json.dumps({"offset": offset.tolist(), "correlation_value": float(correlation_value)})
        else:
            return offset

    def reconcile_image_shape(
        self,
        image_t: np.ndarray,
        reference_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Crop or pad target image to match reference shape."""
        if image_t.shape == reference_shape:
            return image_t

        output = image_t
        for i in range(2):
            if self.image_coordinates_origin == "top_left":
                if output.shape[i] < reference_shape[i]:
                    pad_len = [(0, 0), (0, 0)]
                    pad_len[i] = (0, reference_shape[i] - output.shape[i])
                    output = np.pad(output, pad_len, mode="constant")
                elif output.shape[i] > reference_shape[i]:
                    slicer = [slice(None)] * 2
                    slicer[i] = slice(0, reference_shape[i])
                    output = output[tuple(slicer)]
            elif self.image_coordinates_origin == "center":
                if output.shape[i] < reference_shape[i]:
                    pad_len = [(0, 0), (0, 0)]
                    delta = reference_shape[i] - output.shape[i]
                    pad_len[i] = (delta // 2, delta - delta // 2)
                    output = np.pad(output, pad_len, mode="constant")
                elif output.shape[i] > reference_shape[i]:
                    slicer = [slice(None)] * 2
                    delta = output.shape[i] - reference_shape[i]
                    slicer[i] = slice(delta // 2, delta // 2 + reference_shape[i])
                    output = output[tuple(slicer)]
            else:
                raise ValueError(
                    f"Invalid value for image_coordinates_origin: {self.image_coordinates_origin}"
                )
        return output

    def prepare_image_for_feature_matching(self, image: np.ndarray) -> np.ndarray:
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image, dtype=np.float32)
        return image

    def adjust_points_for_origin(
        self,
        points: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> np.ndarray:
        if self.image_coordinates_origin == "center":
            center = np.array([(image_shape[1] - 1) / 2, (image_shape[0] - 1) / 2])
            return points - center
        if self.image_coordinates_origin == "top_left":
            return points
        raise ValueError(
            f"Invalid value for image_coordinates_origin: {self.image_coordinates_origin}"
        )

    def feature_based_registration(
        self,
        image_t: np.ndarray,
        image_r: np.ndarray,
    ) -> np.ndarray:
        image_t_float = self.prepare_image_for_feature_matching(image_t)
        image_r_float = self.prepare_image_for_feature_matching(image_r)
        sift_t = feature.SIFT()
        sift_r = feature.SIFT()
        sift_t.detect_and_extract(image_t_float)
        sift_r.detect_and_extract(image_r_float)

        descriptors_t = sift_t.descriptors
        descriptors_r = sift_r.descriptors
        keypoints_t = sift_t.keypoints
        keypoints_r = sift_r.keypoints

        if (
            descriptors_t is None
            or descriptors_r is None
            or descriptors_t.size == 0
            or descriptors_r.size == 0
        ):
            raise RuntimeError("SIFT feature detection failed to find descriptors.")

        matches = feature.match_descriptors(
            descriptors_t,
            descriptors_r,
            metric="euclidean",
            cross_check=True,
            max_ratio=0.75,
        )

        if matches.shape[0] < 3:
            raise RuntimeError("Not enough SIFT matches to estimate translation.")

        pts_t = keypoints_t[matches[:, 0]][:, ::-1]
        pts_r = keypoints_r[matches[:, 1]][:, ::-1]
        pts_t = self.adjust_points_for_origin(pts_t, image_t.shape)
        pts_r = self.adjust_points_for_origin(pts_r, image_r.shape)
        deltas = pts_r - pts_t
        offset_x = np.median(deltas[:, 0])
        offset_y = np.median(deltas[:, 1])
        
        # Reverse the offset since the image is registered with the reference image.
        return np.array([-offset_y, -offset_x])
