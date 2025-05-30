from typing import Annotated, Dict, List, Any
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage as ndi

from eaa.tools.base import BaseTool, check, ToolReturnType
import eaa.comms
import eaa.util


logger = logging.getLogger(__name__)


class AcquireImage(BaseTool):
    
    name: str = "acquire_image"
    
    @check
    def __init__(self, show_image_in_real_time: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.show_image_in_real_time = show_image_in_real_time
        self.rt_fig = None
        
        self.exposed_tools: List[Dict[str, Any]] = [
            {
                "name": "acquire_image",
                "function": self.acquire_image,
                "return_type": ToolReturnType.IMAGE_PATH
            }
        ]
        
    def update_real_time_view(self, image: np.ndarray):
        if self.rt_fig is None:
            self.rt_fig, ax = plt.subplots(1, 1, squeeze=True)
        else:
            ax = self.rt_fig.get_axes()[0]
        ax.clear()
        ax.imshow(image)
        plt.draw()
        plt.pause(0.001)  # Small pause to allow GUI to update

    def acquire_image(self, *args, **kwargs):
        raise NotImplementedError


class SimulatedAcquireImage(AcquireImage):
    
    name: str = "simulated_acquire_image"
    
    def __init__(self, whole_image: np.ndarray, return_message: bool = True, *args, **kwargs):
        self.whole_image = whole_image
        self.interpolator = None
        self.return_message = return_message
        self.blur = None
        self.offset = np.array([0, 0])
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
        
    def set_offset(self, offset: np.ndarray):
        self.offset = offset

    def acquire_image(
        self, 
        loc_y: float, 
        loc_x: float, 
        size_y: int, 
        size_x: int, 
    ) -> Annotated[str, "The path to the acquired image."]:
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
        y = np.arange(loc[0] + self.offset[0], loc[0] + size[0] + self.offset[0])
        x = np.arange(loc[1] + self.offset[1], loc[1] + size[1] + self.offset[1])
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
