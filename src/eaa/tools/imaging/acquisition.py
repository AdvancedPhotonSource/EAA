from typing import Annotated, Dict, List, Any
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage as ndi

from eaa.tools.base import BaseTool, check, ToolReturnType
import eaa.comms
import eaa.util
import eaa.maths


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
    
    def __init__(
        self, 
        whole_image: np.ndarray, 
        return_message: bool = True,
        add_axis_ticks: bool = False,
        *args, **kwargs
    ):
        """The simulated acquisition tool.

        Parameters
        ----------
        whole_image : np.ndarray
            A (h, w) numpy array giving the whole image. Images captured
            with specific locations and sizes will be interpolated from this
            whole image.
        return_message : bool, optional
            If True, the tool returns a string giving the path of the acquired
            image saved on the local hard drive. If False, the tool returns a
            numpy array of the acquired image.
        add_axis_ticks : bool, optional
            If True, the tool adds axis ticks to the acquired image that indicate
            the positions.
        """
        self.whole_image = whole_image
        self.interpolator = None
        self.blur = None
        self.offset = np.array([0, 0])
        
        self.return_message = return_message
        self.add_axis_ticks = add_axis_ticks

        super().__init__(*args, **kwargs)
        
        self.exposed_tools.append(
            {
                "name": "scan_line",
                "function": self.scan_line,
                "return_type": ToolReturnType.IMAGE_PATH
            }
        )
                
    def build(self):
        self.build_interpolator()
        
    def build_interpolator(self, *args, **kwargs):
        self.interpolator = scipy.interpolate.RectBivariateSpline(
            np.arange(self.whole_image.shape[0]),
            np.arange(self.whole_image.shape[1]),
            self.whole_image,
        )
        self.line_interpolator = scipy.interpolate.RegularGridInterpolator(
            (
                np.arange(self.whole_image.shape[0]), 
                np.arange(self.whole_image.shape[1])
            ),
            self.whole_image,
        )
        
    def set_blur(self, blur: float):
        """Set the amount of blurring added to the acquired image
        to simulate out of focus effect.

        Parameters
        ----------
        blur : float
            The standard deviation of the Gaussian blur.
        """
        self.blur = blur
        
    def set_offset(self, offset: np.ndarray):
        """Set the offset of the acquired image to simulate drift.

        Parameters
        ----------
        offset : np.ndarray
            The offset of the acquired image. The offset is given as a tuple
            of (y, x) coordinates.
        """
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
            self.save_image_to_temp_dir(
                arr, 
                filename, 
                add_axis_ticks=self.add_axis_ticks,
                x_ticks=x,
                y_ticks=y,
            )
            return f".tmp/{filename}"
        else:
            return arr


    def scan_line(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        scan_step: float,
    ) -> Annotated[str, "The path to the plot of the line scan."]:
        """Scan along a line in the sample. This function
        generates a plot of the line scan, and a Gaussian fit
        of it. The FWHM of the Gaussian fit is annotated on the plot.

        Parameters
        ----------
        start_x, start_y : float
            The starting point of the line scan.
        end_x, end_y : float
            The ending point of the line scan.
        scan_step : float
            The step size of the line scan.

        Returns
        -------
        str
            The path of the plot of the line scan saved in hard drive.
        """
        pt_start = np.array([start_y, start_x])
        pt_end = np.array([end_y, end_x])
        d_tot = np.linalg.norm(pt_end - pt_start)
        ds = np.arange(0, d_tot, scan_step)
        pts = pt_start + ds[:, None] * (pt_end - pt_start) / d_tot
        
        arr = self.line_interpolator(pts).reshape(-1)
        
        if self.blur is not None and self.blur > 0:
            arr = ndi.gaussian_filter(arr, self.blur)
            
        # Fit a Gaussian to the line scan
        a, mu, sigma, c = eaa.maths.fit_gaussian_1d(ds, arr)
        val_gauss = eaa.maths.gaussian_1d(ds, a, mu, sigma, c)
        fwhm = 2.35 * sigma
        
        fig, ax = plt.subplots(1, 1, squeeze=True)
        ax.plot(ds, arr, label="data")
        ax.plot(ds, val_gauss, linestyle="--", label="Gaussian fit")
        ax.text(
            0.05, 
            0.95, 
            f"FWHM = {fwhm:.2f}", 
            transform=ax.transAxes, 
            verticalalignment="top", 
            horizontalalignment="left"
        )
        ax.legend()
        ax.set_xlabel("distance")
        ax.set_ylabel("value")
        ax.set_title("Line scan")
        ax.grid(True)
        
        fname = f".tmp/line_scan_{start_y}_{end_y}_{start_x}_{end_x}_{scan_step}_{eaa.util.get_timestamp()}.png"
        fig.savefig(fname)
        plt.close(fig)
        return fname
