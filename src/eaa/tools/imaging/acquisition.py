from typing import Any

import numpy as np
import scipy.interpolate

from eaa.tools.base import BaseTool


class AcquireImage(BaseTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        pass


class SimulatedAcquireImage(AcquireImage):
    def __init__(self, whole_image: np.ndarray, *args, **kwargs):
        self.whole_image = whole_image
        self.interpolator = None
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
        loc: tuple[float, float], 
        size: tuple[int, int], 
        *args, **kwargs
    ) -> np.ndarray:
        """Acquire an image of a given size from the whole image at a given
        location.

        Parameters
        ----------
        loc : tuple[float, float]
            The top-left corner location of the image to acquire. The location
            can be floating point number, in which case the image will be
            interpolated.
        size : tuple[int, int]
            The size of the image to acquire.

        Returns
        -------
        np.ndarray
            The acquired image.
        """
        y = np.arange(loc[0], loc[0] + size[0])
        x = np.arange(loc[1], loc[1] + size[1])
        return self.interpolator(y, x).reshape(size)

