import logging
from typing import Optional, Union

from PIL import Image

from eaa.task_managers.base import BaseTaskManager
from eaa.tools.base import BaseTool
from eaa.api.llm_config import LLMConfig
from eaa.agents.memory import MemoryManagerConfig
from eaa.image_proc import stitch_images

logger = logging.getLogger(__name__)


class ImagingBaseTaskManager(BaseTaskManager):
        
    assistant_system_message = (
        "You are helping scientists at a microscopy facility to "
        "calibrate their imaging system and set up their experiments. "
        "You are given the tools that adjust the imaging system, move "
        "the sample stage, and acquire images. "
        "When using tools, only make one call at a time. Do not make "
        "multiple calls simultaneously."
    )
    
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        memory_config: Optional[Union[dict, MemoryManagerConfig]] = None,
        tools: list[BaseTool] = (), 
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args, **kwargs
    ) -> None:
        """An agent that searches for the best setup parameters
        for an imaging system.

        Parameters
        ----------
        llm_config : LLMConfig
            The configuration for the LLM.
        tools : list[BaseTool]
            A list of tools provided to the agent.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        build : bool
            Whether to build the internal state of the task manager.
        """        
        super().__init__(
            llm_config=llm_config,
            memory_config=memory_config,
            tools=tools, 
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
            
    def prerun_check(self, *args, **kwargs) -> bool:
        if len(self.agent.tool_manager.tools) == 0:
            logger.warning("No tools registered for the main agent.")
        return super().prerun_check(*args, **kwargs)
    
    @staticmethod
    def add_reference_image_to_images_acquired(
        new_image_path: str, reference_image_path: str
    ) -> str:
        """Add the reference image to the images acquired side-by-side, and return
        the path to the new image.
        """
        new_image = Image.open(new_image_path)
        reference_image = Image.open(reference_image_path)
        stitched_image = stitch_images([new_image, reference_image], gap=0)
        stitched_image.save(new_image_path)
        return new_image_path
        
    def run(self, *args, **kwargs) -> None:
        """Run the task manager."""
        super().run(*args, **kwargs)
