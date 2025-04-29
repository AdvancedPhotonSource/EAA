from typing import Any, Annotated
import logging

import numpy as np
import scipy.interpolate

from eaa.tools.base import BaseTool


logger = logging.getLogger(__name__)

class AcquireImage(BaseTool):
    
    name: str = "acquire_image"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        pass


class SimulatedAcquireImage(AcquireImage):
    
    name: str = "simulated_acquire_image"
    
    def __init__(self, whole_image: np.ndarray, return_message: bool = True, *args, **kwargs):
        self.whole_image = whole_image
        self.interpolator = None
        self.return_message = return_message
        super().__init__(*args, **kwargs)
                
    def build(self):
        self.build_interpolator()
        
    def build_interpolator(self, *args, **kwargs):
        self.interpolator = scipy.interpolate.RectBivariateSpline(
            np.arange(self.whole_image.shape[0]),
            np.arange(self.whole_image.shape[1]),
            self.whole_image,
        )

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
        np.ndarray
            The acquired image.
        """
        loc = [loc_y, loc_x]
        size = [size_y, size_x]
        logger.info(f"Acquiring image of size {size} at location {loc}.")
        y = np.arange(loc[0], loc[0] + size[0])
        x = np.arange(loc[1], loc[1] + size[1])
        arr = self.interpolator(y, x).reshape(size)
        if self.return_message:
            filename = f"image_{loc_y}_{loc_x}_{size_y}_{size_x}.png"
            self.save_image_to_temp_dir(arr, filename)
            return f"<img .tmp/{filename}>"
        else:
            return arr