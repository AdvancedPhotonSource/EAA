import logging
from typing import Optional

from PIL import Image

from eaa_core.api.llm_config import LLMConfig
from eaa_core.api.memory import MemoryManagerConfig
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.tool.base import BaseTool

from eaa_imaging.image_proc import stitch_images

logger = logging.getLogger(__name__)


class ImagingBaseTaskManager(BaseTaskManager):

    def __init__(
        self, 
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        tools: list[BaseTool] = (), 
        session_db_path: Optional[str] = "session.sqlite",
        build: bool = True,
        *args, **kwargs
    ) -> None:
        """An agent that searches for the best setup parameters
        for an imaging system.

        Parameters
        ----------
        llm_config : LLMConfig
            The configuration for the LLM.
        memory_config : MemoryManagerConfig, optional
            Memory configuration forwarded to the agent.
        tools : list[BaseTool]
            A list of tools provided to the agent.
        session_db_path : Optional[str]
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
            session_db_path=session_db_path,
            build=build,
            *args, **kwargs
        )
            
    def prerun_check(self, *args, **kwargs) -> bool:
        if len(self.tool_executor.tools) == 0:
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
        """Run the imaging task manager.

        Raises
        ------
        NotImplementedError
            Always raised because concrete imaging task managers must
            implement their own `run()` method.
        """
        raise NotImplementedError(
            "Concrete imaging task managers must implement `run()`."
        )
