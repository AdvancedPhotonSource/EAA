from typing import Annotated, Optional
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from skimage import filters, measure
from sciagent.tool.base import BaseTool, ToolReturnType, check, tool

from eaa.tool.imaging.acquisition import AcquireImage

logger = logging.getLogger(__name__)


class TestPatternLandmarkFitting(BaseTool):
    """Fit the circular landmark on the right side of an APS-MIC test image.

    The expected landmark is a bright or dark disk-shaped feature that may be
    only partially visible near the right image boundary. The fitting workflow
    is:

    1. Apply Gaussian smoothing and normalize the image to ``[0, 1]``.
    2. Segment the landmark candidate with binary thresholding.
    3. Apply binary erosion followed by binary dilation with a 3x3 structure
       element for 3 iterations to suppress small islands and weak bridges.
    4. Run connected-component analysis and keep the component whose right-most
       pixel has the largest x coordinate.
    5. Extract the boundary pixels of that component, excluding pixels on the
       outer image border, and fit a circle to those arc pixels with RANSAC.

    The returned center is expressed in the coordinate system of the original,
    uncropped image.
    """

    name: str = "test_pattern_landmark_fitting"

    @check
    def __init__(
        self,
        image_acquisition_tool: Optional[AcquireImage] = None,
        zoom: float = 4.0,
        gaussian_sigma_fraction: float = 0.03,
        ransac_residual_threshold_fraction: float = 0.015,
        ransac_max_trials: int = 1000,
        require_approval: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the landmark fitting tool.

        Parameters
        ----------
        image_acquisition_tool : Optional[AcquireImage], optional
            Acquisition tool that provides ``image_k`` when no image is passed
            directly to :meth:`fit_landmark_center`.
        zoom : float, optional
            Zoom factor applied to the image before segmentation and fitting.
            Returned coordinates are converted back to the original image scale.
        gaussian_sigma_fraction : float, optional
            Gaussian blur sigma expressed as a fraction of the original image
            width in pixels.
        ransac_residual_threshold_fraction : float, optional
            RANSAC inlier threshold expressed as a fraction of the cropped
            image size.
        ransac_max_trials : int, optional
            Maximum number of RANSAC iterations used for the circle fit.
        require_approval : bool, optional
            Whether tool execution requires explicit approval in the agent
            framework.
        """
        super().__init__(*args, require_approval=require_approval, **kwargs)

        self.image_acquisition_tool = image_acquisition_tool
        self.zoom = zoom
        self.gaussian_sigma_fraction = gaussian_sigma_fraction
        self.ransac_residual_threshold_fraction = ransac_residual_threshold_fraction
        self.ransac_max_trials = ransac_max_trials
        self.latest_image: Optional[np.ndarray] = None
        self.latest_circle_model: Optional[measure.CircleModel] = None
        self.latest_circle_inliers: Optional[np.ndarray] = None
        self.latest_processed_image: Optional[np.ndarray] = None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert the input to a finite 2D float image.

        Parameters
        ----------
        image : np.ndarray
            Input image. If the input is 3D, the last axis is averaged.

        Returns
        -------
        np.ndarray
            A 2D floating-point image with non-finite values replaced by the
            mean of the finite pixels.
        """
        arr = np.asarray(image, dtype=float)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=-1)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D image, got shape {arr.shape}.")

        if not np.isfinite(arr).all():
            finite_mask = np.isfinite(arr)
            if not finite_mask.any():
                raise ValueError("Input image does not contain any finite pixels.")
            arr = arr.copy()
            arr[~finite_mask] = float(np.mean(arr[finite_mask]))
        return arr

    def get_input_image(self, image: Optional[np.ndarray]) -> np.ndarray:
        """Return the image to process.

        Parameters
        ----------
        image : Optional[np.ndarray]
            Explicit image to fit. If omitted, ``image_acquisition_tool.image_k``
            is used.

        Returns
        -------
        np.ndarray
            Preprocessed 2D image.
        """
        if image is not None:
            return self.preprocess_image(image)
        if self.image_acquisition_tool is None or self.image_acquisition_tool.image_k is None:
            raise ValueError(
                "No image was provided and image_acquisition_tool.image_k is not available."
            )
        return self.preprocess_image(self.image_acquisition_tool.image_k)

    def zoom_image(self, image: np.ndarray) -> np.ndarray:
        """Zoom the image before processing.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed 2D image in original coordinates.

        Returns
        -------
        np.ndarray
            Image in the processing scale.
        """
        if self.zoom <= 0:
            raise ValueError("zoom must be positive.")
        if self.zoom == 1.0:
            return image
        return ndi.zoom(image, zoom=self.zoom, order=1, mode="nearest")

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Robustly normalize an image to ``[0, 1]`` using percentiles."""
        lo = float(np.percentile(image, 1))
        hi = float(np.percentile(image, 99))
        if hi <= lo:
            lo = float(np.min(image))
            hi = float(np.max(image))
        if hi <= lo:
            return np.zeros_like(image, dtype=float)
        return np.clip((image - lo) / (hi - lo), 0.0, 1.0)

    def detect_edge_points(self, image: np.ndarray) -> np.ndarray:
        """Detect candidate circle-arc points from the segmented landmark.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed 2D image in original coordinates.

        Returns
        -------
        np.ndarray
            Boundary points as ``(x, y)`` pairs in original image coordinates.
        """
        sigma = max(0.5, self.gaussian_sigma_fraction * image.shape[1])
        blurred = ndi.gaussian_filter(image, sigma=sigma, mode="nearest")
        normalized = self.normalize_image(blurred)
        threshold = filters.threshold_otsu(normalized)
        binary = normalized >= threshold

        structure = np.ones((3, 3), dtype=bool)
        # binary = ndi.binary_erosion(binary, structure=structure, iterations=1)
        # binary = ndi.binary_dilation(binary, structure=structure, iterations=1)
        # binary = ndi.binary_closing(binary, structure=structure, iterations=1)

        labels, num_labels = ndi.label(binary, structure=structure)
        if num_labels == 0:
            raise ValueError("No connected component was found in the segmented image.")

        selected_label = None
        rightmost_x = -1
        for label_id in range(1, num_labels + 1):
            label_y, label_x = np.nonzero(labels == label_id)
            if label_x.size == 0:
                continue
            label_rightmost_x = int(np.max(label_x))
            if label_rightmost_x > rightmost_x:
                rightmost_x = label_rightmost_x
                selected_label = label_id

        if selected_label is None:
            raise ValueError("Failed to select a segmented landmark component.")

        component = labels == selected_label
        boundary = component & ~ndi.binary_erosion(component, structure=structure, iterations=1)
        valid_support = np.zeros_like(component, dtype=bool)
        valid_support[1:-1, 1:-1] = True
        boundary &= valid_support

        edge_y, edge_x = np.nonzero(boundary)
        if edge_x.size < 3:
            raise ValueError("Segmented landmark boundary has fewer than 3 candidate points.")

        points = np.column_stack((edge_x.astype(float), edge_y.astype(float)))
        return points

    def fit_circle(
        self,
        points: np.ndarray,
        cropped_shape: tuple[int, int],
    ) -> tuple[measure.CircleModel, np.ndarray]:
        """Fit a circle with RANSAC and return its parameters.

        Parameters
        ----------
        points : np.ndarray
            Candidate edge points as ``(x, y)`` pairs in the original image
            frame.
        cropped_shape : tuple[int, int]
            Shape of the image as ``(height, width)``.

        Returns
        -------
        tuple[measure.CircleModel, np.ndarray]
            The fitted circle model and its boolean inlier mask.
        """
        residual_threshold = max(
            1.0,
            self.ransac_residual_threshold_fraction * max(cropped_shape),
        )
        model, inliers = measure.ransac(
            points,
            measure.CircleModel,
            min_samples=3,
            residual_threshold=residual_threshold,
            max_trials=self.ransac_max_trials,
        )
        if model is None or inliers is None or int(np.count_nonzero(inliers)) < 3:
            raise ValueError("RANSAC could not find a valid circle from the detected edges.")

        center_x, center_y, radius = model.params
        logger.debug(
            "Circle fit: center=(%.3f, %.3f), radius=%.3f, inliers=%d/%d",
            center_x,
            center_y,
            radius,
            int(np.count_nonzero(inliers)),
            len(points),
        )
        return model, inliers

    def plot_last_fit(self) -> plt.Figure:
        """Plot the most recent image with the fitted circle overlaid.

        Returns
        -------
        matplotlib.figure.Figure
            Figure showing the stored image and the last fitted circle.
        """
        if self.latest_image is None or self.latest_circle_model is None:
            raise ValueError("No fitted landmark is available. Call fit_landmark_center first.")

        center_x, center_y, radius = self.latest_circle_model.params
        theta = np.linspace(0.0, 2.0 * np.pi, 512)
        circle_x = center_x + radius * np.cos(theta)
        circle_y = center_y + radius * np.sin(theta)

        fig, ax = plt.subplots(1, 1, squeeze=True)
        ax.imshow(self.latest_image, cmap="viridis", origin="upper")
        ax.plot(circle_x, circle_y, color="cyan", linewidth=1.5)
        ax.scatter([center_x], [center_y], color="red", s=30)
        ax.set_title("Landmark Circle Fit")
        ax.set_xlim(-0.5, self.latest_image.shape[1] - 0.5)
        ax.set_ylim(self.latest_image.shape[0] - 0.5, -0.5)
        return fig

    @tool(name="fit_landmark_center", return_type=ToolReturnType.LIST)
    def fit_landmark_center(
        self,
        image: Annotated[
            Optional[np.ndarray],
            "Optional 2D image array. When omitted, the tool uses image_acquisition_tool.image_k.",
        ] = None,
    ) -> Annotated[
        list[float],
        "The fitted landmark center as [center_y, center_x] in the coordinates of the original image.",
    ]:
        """Detect the right-side disk feature and return its center.

        Parameters
        ----------
        image : Optional[np.ndarray], optional
            Explicit image array to process. If omitted, the latest image from
            ``image_acquisition_tool`` is used.

        Returns
        -------
        list[float]
            Landmark center as ``[center_y, center_x]`` in the coordinates of
            the original image.
        """
        image_arr = self.get_input_image(image)
        processed_image = self.zoom_image(image_arr)
        points = self.detect_edge_points(processed_image)
        model, inliers = self.fit_circle(
            points,
            cropped_shape=processed_image.shape,
        )
        center_x, center_y, radius = model.params
        circle_model = measure.CircleModel()
        circle_model.params = (
            float(center_x / self.zoom),
            float(center_y / self.zoom),
            float(radius / self.zoom),
        )
        self.latest_image = image_arr
        self.latest_processed_image = processed_image
        self.latest_circle_model = circle_model
        self.latest_circle_inliers = inliers
        return [float(center_y / self.zoom), float(center_x / self.zoom)]
