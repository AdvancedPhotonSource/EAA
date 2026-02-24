from typing import Annotated, Any, List, Literal, Optional, Tuple
import logging
import re
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from sciagent.tool.base import BaseTool, check, ToolReturnType, tool
from sciagent.task_manager.base import BaseTaskManager
from sciagent.message_proc import generate_openai_message
from sciagent.skill import SkillMetadata
from sciagent.api.llm_config import LLMConfig
from sciagent.api.memory import MemoryManagerConfig

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
        llm_config: Optional[LLMConfig] = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        reference_image: np.ndarray = None,
        reference_pixel_size: float = 1.0,
        image_coordinates_origin: Literal["top_left", "center"] = "top_left",
        registration_method: Literal["phase_correlation", "sift", "mutual_information", "llm"] = "phase_correlation",
        log_scale: bool = False,
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
            finds the translation to apply to the test image so that it aligns
            with the reference image where the values are the same for the same coordinates
            (i, j). When the origin defining the coordinates is different, the
            test image is cropped/padded differently when its size does not match.
            When this argument is set to "center", the test image is padded/cropped
            centrally. When it is set to "top_left", the test image is on the bottom
            and right sides.
        registration_method : Literal["phase_correlation", "sift", "mutual_information"], optional
            The method used to estimate translational offsets. "phase_correlation"
            uses phase correlation, "sift" uses feature matching, and
            "mutual_information" uses pyramid-based normalized mutual information.
        log_scale : bool, optional
            If True, images are transformed as `log10(x + 1)` before registration.
        """
        super().__init__(*args, require_approval=require_approval, **kwargs)

        self.image_acquisition_tool = image_acquisition_tool
        self.llm_config = llm_config
        self.memory_config = memory_config
        self.reference_image = reference_image
        self.reference_pixel_size = reference_pixel_size
        self.image_coordinates_origin = image_coordinates_origin
        self.registration_method = registration_method
        self.log_scale = log_scale

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
        image[np.isnan(image)] = np.mean(image)
        if self.log_scale:
            image = np.log10(image + 1)
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
        "The translational offset [dy (vertical), dx (horizontal)] to apply to the latest "
        "acquired image so it aligns with the reference image. Positive y means shifting the "
        "latest image downward; positive x means shifting it rightward. The returned values are "
        "in physical units, i.e., pixel size is already accounted for.",
    ]:
        """
        Register the latest image collected by the image acquisition tool
        and the reference image.
        """
        image_t, image_r, psize_t, psize_r = self.get_registration_inputs(register_with)
        
        offset = self.register_images(image_t, image_r, psize_t, psize_r)
        return offset

    def get_registration_inputs(
        self,
        register_with: Literal["previous", "first", "reference"],
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
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
        return image_t, image_r, float(psize_t), float(psize_r)

    @tool(name="apply_and_view_offset", return_type=ToolReturnType.IMAGE_PATH)
    def apply_and_view_offset(
        self,
        register_with: Annotated[
            Literal["previous", "first", "reference"],
            "The image to register the latest image with. "
            "Can be 'previous', 'first', or 'reference'.",
        ],
        fractional_offset_y: Annotated[
            float,
            "Fractional y-shift (down is positive) relative to image height.",
        ],
        fractional_offset_x: Annotated[
            float,
            "Fractional x-shift (right is positive) relative to image width.",
        ],
    ) -> Annotated[str, "Path to side-by-side plot of reference image and shifted current image."]:
        """Apply a fractional offset to the latest (moving) image and
        view it side by side with the reference image.
        """
        image_t, image_r, _, _ = self.get_registration_inputs(register_with)
        shift_pixels = np.array(
            [
                fractional_offset_y * float(image_r.shape[0]),
                fractional_offset_x * float(image_r.shape[1]),
            ],
            dtype=float,
        )
        shifted_image_t = ndi.shift(
            image_t,
            shift=shift_pixels,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        fig = self.build_registration_pair_figure(
            image_r,
            shifted_image_t,
            images_are_processed=True,
        )
        return BaseTool.save_image_to_temp_dir(
            fig=fig,
            filename="llm_registration_offset_check.png",
            add_timestamp=True,
        )

    def parse_llm_shift(self, response_text: str) -> np.ndarray:
        content = response_text.strip()
        match = re.search(r"^\s*([^,]+)\s*,\s*([^,]+)\s*$", content)
        if match is None:
            match = re.search(r"([+-]?(?:\d+\.?\d*|\.\d+|nan))\s*,\s*([+-]?(?:\d+\.?\d*|\.\d+|nan))", content, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Unable to parse registration shift from response: {response_text}")
        tokens = [match.group(1).strip().lower(), match.group(2).strip().lower()]
        vals: list[float] = []
        for token in tokens:
            if token == "nan":
                vals.append(float("nan"))
            else:
                vals.append(float(token))
        return np.array(vals, dtype=float)

    def build_registration_pair_figure(
        self,
        image_r: np.ndarray,
        image_t: np.ndarray,
        images_are_processed: bool = False,
    ):
        if not images_are_processed:
            image_r = self.process_image(image_r)
            image_t = self.process_image(image_t)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_r, cmap="inferno", origin="upper")
        axes[0].set_title("Previous / Reference")
        axes[0].set_axis_off()
        axes[1].imshow(image_t, cmap="inferno", origin="upper")
        axes[1].set_title("Current / Target")
        axes[1].set_axis_off()
        fig.tight_layout()
        return fig

    def register_images_llm(self, image_t: np.ndarray, image_r: np.ndarray) -> np.ndarray:
        if self.llm_config is None:
            logger.warning("`llm_config` is not set for LLM image registration. Returning NaN offset.")
            return np.array([np.nan, np.nan], dtype=float)

        skill_path = (
            Path(__file__).resolve().parents[2]
            / "private_skills"
            / "image-registration"
            / "SKILL.md"
        )
        if not skill_path.exists():
            raise FileNotFoundError(f"Registration skill file not found: {skill_path}")
        skill_text = skill_path.read_text(encoding="utf-8")

        fig = self.build_registration_pair_figure(
            image_r,
            image_t,
            images_are_processed=True,
        )
        image_path = BaseTool.save_image_to_temp_dir(
            fig=fig,
            filename="llm_registration_pair.png",
            add_timestamp=True,
        )

        registration_task = BaseTaskManager(
            llm_config=self.llm_config,
            tools=[self],
            use_coding_tools=False,
            build=True,
        )
        skill_tool_name = "skill-image-registration"
        registration_task.skill_catalog = [
            SkillMetadata(
                name="image-registration",
                description="Instructions for image registration.",
                tool_name=skill_tool_name,
                path=str(skill_path.parent),
            )
        ]
        registration_task._inject_skill_doc_messages_to_context(
            tool_response={
                "content": {
                    "path": str(skill_path.parent),
                    "files": {"SKILL.md": skill_text},
                }
            },
            tool_call_info={"function": {"name": skill_tool_name}},
        )
        registration_task.run_conversation(
            message=generate_openai_message(
                content=(
                    "Estimate the translational shift to apply to the current/test image "
                    "so it aligns with the previous/reference image, and return only "
                    "'<shift_y>, <shift_x>'. "
                    "Before final answer, call apply_and_view_offset to verify your proposed offset. "
                    "Use register_with='previous' for verification."
                ),
                role="user",
                image_path=image_path,
            ),
            termination_behavior="return",
        )
        response_text = None
        for message in reversed(registration_task.context):
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                response_text = message["content"]
                break
        if response_text is None:
            raise RuntimeError("No assistant response found in LLM image registration context.")

        shift_fraction = self.parse_llm_shift(response_text)
        if np.any(np.isnan(shift_fraction)):
            return np.array([np.nan, np.nan], dtype=float)
        return np.array([
            shift_fraction[0] * float(image_r.shape[0]),
            shift_fraction[1] * float(image_r.shape[1]),
        ], dtype=float)

    def register_images(
        self, 
        image_t: np.ndarray, 
        image_r: np.ndarray, 
        psize_t: float, 
        psize_r: float,
        registration_method: Optional[Literal["phase_correlation", "sift", "mutual_information", "llm"]] = None,
        registration_algorithm_kwargs: Optional[dict[str, Any]] = None,
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
        registration_method : Optional[Literal["phase_correlation", "sift", "mutual_information"]], optional
            Overrides the default registration method for this call. 
        registration_algorithm_kwargs : Optional[dict[str, Any]], optional
            Optional keyword arguments forwarded to the selected registration
            algorithm. Supported keys and defaults depend on `registration_method`:

            - `registration_method="phase_correlation"`:
              - `use_hanning_window` (bool, default: `True`)
              - `upsample_factor` (int, default: `1`)

            - `registration_method="mutual_information"`:
              - `pyramid_levels` (tuple[int, ...], default: `(4, 2, 1)`)
              - `bins` (int, default: `64`)
              - `sample_frac` (float, default: `0.2`)
              - `smooth_sigmas` (Optional[dict[int, float]], default: `None`)
              - `optimizer` (Literal["powell", "nelder-mead"], default: `"nelder-mead"`)
              - `max_iter` (int, default: `60`)
              - `tol` (float, default: `1e-4`)

            - `registration_method="sift"` or `"llm"`:
              - No algorithm kwargs are currently supported; pass `None` or `{}`.

        Returns
        -------
        np.ndarray | str
            The translation offset (dy, dx) to apply to the target image so it aligns
            with the reference image. Positive y means shifting the test image
            downward; positive x means shifting the target image rightward. Returned
            values are in physical units, i.e., pixel size is already accounted for.
        """
        method = registration_method or self.registration_method
        algorithm_kwargs = dict(registration_algorithm_kwargs or {})

        # Handle pixel size and image size differences
        if psize_t != psize_r:
            # Resize the target image to have the same pixel size as the reference image
            image_t = ndi.zoom(image_t, psize_t / psize_r)
        
        if method in {"phase_correlation", "mutual_information"}:
            image_t = self.reconcile_image_shape(image_t, image_r.shape)

        if method == "phase_correlation":
            phase_kwargs = {"use_hanning_window": True}
            phase_kwargs.update(algorithm_kwargs)
            offset = phase_cross_correlation(
                image_t,
                image_r,
                **phase_kwargs,
            )
        elif method == "mutual_information":
            mi_kwargs = {
                "pyramid_levels": (4, 2, 1),
                "bins": 64,
                "sample_frac": 0.2,
                "optimizer": "nelder-mead",
                "max_iter": 60,
                "tol": 1e-4,
            }
            mi_kwargs.update(algorithm_kwargs)
            offset = translation_nmi_registration(
                moving=image_t,
                ref=image_r,
                **mi_kwargs,
            )
        elif method == "sift":
            if len(algorithm_kwargs) > 0:
                raise ValueError(
                    "`registration_algorithm_kwargs` is not supported for "
                    "registration_method='sift'."
                )
            offset = self.feature_based_registration(image_t, image_r)
        elif method == "llm":
            if len(algorithm_kwargs) > 0:
                raise ValueError(
                    "`registration_algorithm_kwargs` is not supported for "
                    "registration_method='llm'."
                )
            offset = self.register_images_llm(image_t=image_t, image_r=image_r)
        else:
            raise ValueError(f"Invalid registration method: {method}")
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
        return np.array([offset_y, offset_x])
