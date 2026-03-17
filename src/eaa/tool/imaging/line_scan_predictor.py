from typing import Annotated
import io
import json
import logging

import numpy as np
import requests
import tifffile
from eaa.core.tooling.base import BaseTool, check, tool

from eaa.tool.imaging.acquisition import AcquireImage

logger = logging.getLogger(__name__)


class LineScanPredictor(BaseTool):
    """Tool that queries a LineScanPredictor server to predict the optimal
    line scan position based on a reference image and the current image.
    """

    name: str = "line_scan_predictor"

    @check
    def __init__(
        self,
        server_url: str,
        image_acquisition_tool: AcquireImage,
        require_approval: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the line scan predictor tool.

        Parameters
        ----------
        server_url : str
            The base URL of the LineScanPredictor FastAPI server,
            e.g. ``"http://localhost:8090"``.
        image_acquisition_tool : AcquireImage
            The image acquisition tool instance. Must be the same object
            used by the task manager so that its image buffers and call
            history reflect the current state.
        """
        super().__init__(*args, require_approval=require_approval, **kwargs)
        self.server_url = server_url.rstrip("/")
        self.image_acquisition_tool = image_acquisition_tool

    @staticmethod
    def _encode_as_tiff(image: np.ndarray) -> bytes:
        buf = io.BytesIO()
        tifffile.imwrite(buf, image.astype(np.float32))
        return buf.getvalue()

    @tool(name="predict_line_scan_position")
    def predict_line_scan_position(
        self,
    ) -> Annotated[
        str,
        "JSON-encoded predicted line scan center position "
        "{'center_y': float, 'center_x': float} in the physical coordinate "
        "system of the current image.",
    ]:
        """Predict the optimal line scan center position for the current image.

        Uses the first acquired image (image_0) as the reference image and
        the first recorded line scan position as the reference position.
        Queries the LineScanPredictor server and returns the predicted center
        converted back to the physical coordinate system of the current image.

        Returns
        -------
        str
            JSON string with keys ``center_y`` and ``center_x`` giving the
            predicted line scan center in the physical coordinate system of
            the current image (same units as the image acquisition coordinates).
        """
        acq = self.image_acquisition_tool

        if acq.image_0 is None or acq.image_k is None:
            raise RuntimeError(
                "Image buffers are not populated. Acquire at least one image "
                "before calling predict_line_scan_position."
            )
        if not acq.line_scan_call_history:
            raise RuntimeError(
                "No line scan history found. Perform at least one line scan "
                "before calling predict_line_scan_position."
            )
        if not acq.image_acquisition_call_history:
            raise RuntimeError("No image acquisition history found.")

        # --- Reference image info (from the first acquisition) ---
        ref_img_info = acq.image_acquisition_call_history[0]
        ref_line_info = acq.line_scan_call_history[0]

        # Compute the center of the reference line scan in physical coordinates.
        ref_center_x_phys = ref_line_info["x_center"]
        ref_center_y_phys = ref_line_info["y_center"]

        # Convert to fractions of the reference image dimensions.
        ref_center_x_frac = (
            (ref_center_x_phys - (ref_img_info["x_center"] - ref_img_info["size_x"] / 2))
            / ref_img_info["size_x"]
        )
        ref_center_y_frac = (
            (ref_center_y_phys - (ref_img_info["y_center"] - ref_img_info["size_y"] / 2))
            / ref_img_info["size_y"]
        )

        logger.debug(
            "Reference line scan center: phys=(%.3f, %.3f), frac=(%.3f, %.3f)",
            ref_center_y_phys, ref_center_x_phys,
            ref_center_y_frac, ref_center_x_frac,
        )

        # --- Query the server ---
        ref_tiff = self._encode_as_tiff(acq.image_0)
        test_tiff = self._encode_as_tiff(acq.image_k)

        response = requests.post(
            f"{self.server_url}/predict",
            data={
                "ref_center_y": ref_center_y_frac,
                "ref_center_x": ref_center_x_frac,
            },
            files={
                "ref_image": ("ref_image.tif", ref_tiff, "image/tiff"),
                "test_image": ("test_image.tif", test_tiff, "image/tiff"),
            },
        )
        response.raise_for_status()
        result = response.json()

        pred_center_y_frac = result["pred_center_y"]
        pred_center_x_frac = result["pred_center_x"]

        # --- Convert predicted fractions back to physical coordinates ---
        # The fractions are relative to the current (test) image dimensions.
        cur_img_info = acq.image_acquisition_call_history[-1]
        pred_center_y_phys = (
            cur_img_info["y_center"] - cur_img_info["size_y"] / 2
            + pred_center_y_frac * cur_img_info["size_y"]
        )
        pred_center_x_phys = (
            cur_img_info["x_center"] - cur_img_info["size_x"] / 2
            + pred_center_x_frac * cur_img_info["size_x"]
        )

        logger.debug(
            "Predicted line scan center: frac=(%.3f, %.3f), phys=(%.3f, %.3f)",
            pred_center_y_frac, pred_center_x_frac,
            pred_center_y_phys, pred_center_x_phys,
        )

        return json.dumps({"center_y": pred_center_y_phys, "center_x": pred_center_x_phys})
