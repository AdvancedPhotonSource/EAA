from typing import Annotated
import logging
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage as ndi
import autogen
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

from eaa.tools.base import BaseTool
import eaa.comms
import eaa.util

logger = logging.getLogger(__name__)


class AcquireImage(BaseTool):
    
    name: str = "acquire_image"
    
    def __init__(self, show_image_in_real_time: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.show_image_in_real_time = show_image_in_real_time
        self.rt_fig = None
        
    def update_real_time_view(self, image: np.ndarray):
        if self.rt_fig is None:
            self.rt_fig, ax = plt.subplots(1, 1, squeeze=True)
        else:
            ax = self.rt_fig.get_axes()[0]
        ax.clear()
        ax.imshow(image)
        plt.draw()
        plt.pause(0.001)  # Small pause to allow GUI to update

    def __call__(self, *args, **kwargs):
        pass
    
    
class BlueSkyAcquireImage(AcquireImage):
    
    name: str = "bluesky_acquire_image"
    
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BlueSkyAcquireImage is not implemented.")


class SimulatedAcquireImage(AcquireImage):
    
    name: str = "simulated_acquire_image"
    
    def __init__(self, whole_image: np.ndarray, return_message: bool = True, *args, **kwargs):
        self.whole_image = whole_image
        self.interpolator = None
        self.return_message = return_message
        self.blur = None
        super().__init__(*args, **kwargs)
                
    def build(self):
        self.build_interpolator()
        
    def build_interpolator(self, *args, **kwargs):
        self.interpolator = scipy.interpolate.RectBivariateSpline(
            np.arange(self.whole_image.shape[0]),
            np.arange(self.whole_image.shape[1]),
            self.whole_image,
        )
        
    def set_blur(self, blur: float):
        self.blur = blur

    def __call__(
        self, 
        loc_y: float, 
        loc_x: float, 
        size_y: int, 
        size_x: int, 
    ) -> Annotated[str, "The acquired image path."]:
        """Acquire an image of a given size from the whole image at a given
        location.

        Parameters
        ----------
        loc_y, loc_x : float
            The top-left corner location of the image to acquire. The location
            can be floating point number, in which case the image will be
            interpolated.
        size_y, size_x : int
            The size of the image to acquire.

        Returns
        -------
        str
            The path of the acquired image saved in hard drive.
        """
        loc = [loc_y, loc_x]
        size = [size_y, size_x]
        logger.info(f"Acquiring image of size {size} at location {loc}.")
        y = np.arange(loc[0], loc[0] + size[0])
        x = np.arange(loc[1], loc[1] + size[1])
        arr = self.interpolator(y, x).reshape(size)
        
        if self.blur is not None and self.blur > 0:
            arr = ndi.gaussian_filter(arr, self.blur)
        
        if self.show_image_in_real_time:
            self.update_real_time_view(arr)
        if self.return_message:
            filename = f"image_{loc_y}_{loc_x}_{size_y}_{size_x}_{eaa.util.get_timestamp()}.png"
            self.save_image_to_temp_dir(arr, filename)
            return f".tmp/{filename}"
        else:
            return arr
        

class AcquireImageAndAnalyze(BaseTool):
    
    name: str = "acquire_image_and_analyze"
    
    def __init__(
        self, 
        acquisition_tool: AcquireImage, 
        model_name: str = "openai/gpt-4o-2024-11-20",
        model_base_url: str = "https://openrouter.ai/api/v1",
        *args, **kwargs
    ):
        self.acquisition_tool = acquisition_tool
        
        self.model_name = model_name
        self.model_base_url = model_base_url
        self.agent = None
        self.user_proxy = None
        super().__init__(*args, **kwargs)
        
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.build_agent()
    def build_agent(self, *args, **kwargs):
        self.agent = MultimodalConversableAgent(
            llm_config={
                "model": self.model_name,
                "api_key": eaa.comms.get_api_key(self.model_name, self.model_base_url),
                "base_url": self.model_base_url,
            },
            name="image_analysis_agent",
            system_message=dedent("""\
                Your job is to analyze a given image to identify the objects
                or features in it.\
                """
            ),
        )
        
        self.user_proxy = autogen.UserProxyAgent(
            name="acquire_image_and_analyze_user_proxy",
            system_message="A human admin.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={
                "use_docker": False
            },
        )
        
    def __call__(
        self, 
        loc_y: float, 
        loc_x: float, 
        size_y: int, 
        size_x: int, 
    ) -> Annotated[str, "A description of the object contained in the image"]:
        """Acquire an image of a given size from the whole image at a given
        location, then analyze it to identify the objects or features in the
        image.

        Parameters
        ----------
        loc_y, loc_x : float
            The top-left corner location of the image to acquire. The location
            can be floating point number, in which case the image will be
            interpolated.
        size_y, size_x : int
            The size of the image to acquire.

        Returns
        -------
        str
            A description of the objects contained in the image.
        """
        image_path = self.acquisition_tool(loc_y, loc_x, size_y, size_x)
        message = dedent(
            f"""\
            Please describe the objects or features in the image. For each
            feature, describe its location, and particularly note if
            it is in the center of the image. If
            you cannot see anything, just say the image is empty.
            <img {image_path}>\
            """
        )
        self.user_proxy.initiate_chat(
            self.agent,
            message=message,
        )
        return self.user_proxy.last_message(self.agent)["content"]
