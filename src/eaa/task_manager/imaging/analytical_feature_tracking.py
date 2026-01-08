from typing import Optional, Tuple
import copy
import json
import logging

import numpy as np
from sciagent.api.llm_config import LLMConfig
from sciagent.api.memory import MemoryManagerConfig

from eaa.tool.imaging.acquisition import AcquireImage
from eaa.tool.imaging.registration import ImageRegistration
from eaa.task_manager.imaging.base import ImagingBaseTaskManager

logger = logging.getLogger(__name__)


class AnalyticalFeatureTrackingTaskManager(ImagingBaseTaskManager):
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        image_acquisition_tool: AcquireImage = None,
        message_db_path: Optional[str] = None,
        build: bool = True,
        image_acquisition_tool_x_coordinate_args: Tuple[str, ...] = ("x_center",),
        image_acquisition_tool_y_coordinate_args: Tuple[str, ...] = ("y_center",),
        *args, **kwargs
    ) -> None:
        """Move the FOV in a spiral pattern to look for a feature in a
        reference image.

        Parameters
        ----------
        llm_config : LLMConfig
            The configuration for the LLM.
        memory_config : MemoryManagerConfig, optional
            Memory configuration forwarded to the agent.
        image_acquisition_tool : AcquireImage
            The tool to use to acquire images.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        build : bool
            Whether to build the internal state of the task manager.
        image_acquisition_tool_x_coordinate_args: Tuple[str, ...]
            The names of the arguments of the image acquisition tool that specify x-coordinates.
        image_acquisition_tool_y_coordinate_args: Tuple[str, ...]
            The names of the arguments of the image acquisition tool that specify y-coordinates.
        """
        if image_acquisition_tool is None:
            raise ValueError("image_acquisition_tool must be provided.")
        
        self.image_acquisition_tool = image_acquisition_tool
        self.image_registration_tool = self.create_image_registration_tool(image_acquisition_tool)
        
        self.image_acquisition_tool_x_coordinate_args = image_acquisition_tool_x_coordinate_args
        self.image_acquisition_tool_y_coordinate_args = image_acquisition_tool_y_coordinate_args
        
        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            tools=[], 
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
        
    def create_image_registration_tool(self, acquisition_tool: AcquireImage):
        image_registration_tool = ImageRegistration(
            image_acquisition_tool=acquisition_tool,
            reference_image=None,
            reference_pixel_size=1.0,
            image_coordinates_origin="top_left",
        )
        return image_registration_tool
    
    @staticmethod
    def get_position_deltas(idx: int, step_size: Tuple[float, float]) -> Tuple[float, float]:
        """Get the delta of y/x positions of the FOV relative to the initial position
        given the index of the current FOV in the spiral pattern.

        Parameters
        ----------
        idx : int
            The index of the current FOV in the spiral pattern.
        step_size : Tuple[float, float]
            The step size of the spiral pattern in y/x directions.

        Returns
        -------
        Tuple[float, float]
            The delta of y/x positions of the FOV relative to the initial position.
        """
        if idx == 0:
            return 0, 0
        
        # Determine the "radius", or the layer of the loop in the spiral pattern.
        r = 1
        while idx >= (2 * r + 1) ** 2:
            r += 1
        idx_current_loop = idx - (2 * (r - 1) + 1) ** 2
        side_len = 2 * r
        
        # Top edge (moving left to right, includes top-right corner)
        if idx_current_loop < side_len:
            iy = -r
            ix = -r + 1 + idx_current_loop
        # Right edge (moving top to bottom, includes bottom-right corner)
        elif idx_current_loop < 2 * side_len:
            iy = -r + 1 + (idx_current_loop - side_len)
            ix = r
        # Bottom edge (moving right to left, includes bottom-left corner)
        elif idx_current_loop < 3 * side_len:
            iy = r
            ix = r - 1 - (idx_current_loop - 2 * side_len)
        # Left edge (moving bottom to top, includes top-left corner)
        elif idx_current_loop < 4 * side_len:
            iy = r - 1 - (idx_current_loop - 3 * side_len)
            ix = -r
        else:
            raise ValueError(f"Invalid index: {idx}")
        return iy * step_size[0], ix * step_size[1]
    
    def update_kwargs_buffers(
        self,
        current_acquisition_kwargs: dict,
        y_delta: float,
        x_delta: float,
    ):
        for arg in self.image_acquisition_tool_y_coordinate_args:
            current_acquisition_kwargs[arg] += y_delta
        for arg in self.image_acquisition_tool_x_coordinate_args:
            current_acquisition_kwargs[arg] += x_delta
        return current_acquisition_kwargs
    
    def run(
        self,
        current_acquisition_kwargs: dict,
        reference_image: np.ndarray,
        step_size: Tuple[float, float],
        reference_image_pixel_size: float = 1.0,
        n_max_rounds: int = 20,
        correlation_threshold: float = 0.7,
    ) -> np.ndarray:
        """Run the feature tracking task manager.
        
        Parameters
        ----------
        current_acquisition_kwargs: dict
            The current kwargs of the image acquisition tool.
        reference_image: np.ndarray
            A 2D numpy array of the reference image to look for the feature in.
        step_size: Tuple[float, float]
            The step size of the spiral pattern in y/x directions.
        n_max_rounds: int
            The maximum number of rounds to run the feature tracking task manager.
        correlation_threshold: float
            The threshold of the correlation value to consider the feature present
            in the current image.
            
        Returns
        -------
        np.ndarray
            Offset in y and x. If these offsets are added to the initial positions
            in `initial_acquisition_kwargs`, the FOV should be aligned with the reference
            image.
        """
        initial_acquisition_kwargs = copy.deepcopy(current_acquisition_kwargs)
        self.image_registration_tool.set_reference_image(
            reference_image, reference_pixel_size=reference_image_pixel_size
        )
        
        for i in range(n_max_rounds):
            y_delta, x_delta = self.get_position_deltas(i, step_size)
            acquisition_kwargs = self.update_kwargs_buffers(
                copy.deepcopy(initial_acquisition_kwargs), y_delta, x_delta
            )
            current_image_path = self.image_acquisition_tool.acquire_image(**acquisition_kwargs)
            image = self.image_acquisition_tool.image_k
            
            # Get offset with windowing
            res = json.loads(
                self.image_registration_tool.register_images(
                    image, 
                    reference_image, 
                    psize_t=self.image_acquisition_tool.psize_k,
                    psize_r=self.image_registration_tool.reference_pixel_size,
                    return_correlation_value=True,
                    use_hanning_window=True
                )
            )
            offset = res["offset"]

            # Get correlation value without windowing
            res = json.loads(
                self.image_registration_tool.register_images(
                    image, 
                    reference_image, 
                    psize_t=self.image_acquisition_tool.psize_k,
                    psize_r=self.image_registration_tool.reference_pixel_size,
                    return_correlation_value=True,
                    use_hanning_window=False
                )
            )
            correlation_value = res["correlation_value"]
            logger.info(f"Correlation value: {correlation_value}")
            self.record_system_message(
                f"Correlation value: {correlation_value}",
                image_path=current_image_path,
            )
            if correlation_value > correlation_threshold:
                break
        return np.array([y_delta, x_delta]) + offset
