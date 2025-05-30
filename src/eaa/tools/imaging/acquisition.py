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
    
    
class BlueSkyAcquireImage(AcquireImage):

    from bluesky.run_engine import RunEngine
    from mic_common.devices.save_data import SaveDataMic
    from typing import Callable
    
    name: str = "bluesky_acquire_image"
    RE: RunEngine = None
    scanplan: Callable = None
    savedata: SaveDataMic = None
    
    def __init__(self):
        try:
            from .bluesky_init import RE, step2d, get_control_components
            self.RE = RE
            self.scanplan = step2d
            self.savedata = get_control_components("savedata")
        except ImportError:
            raise ImportError("Bluesky control initialization failed. " +
                            "Please check that the bluesky-mic package is installed" + 
                            "and the motors can only be reached from private subnet computers.")
        
    def __call__(
        self,
        samplename="smp1",
        width=0,
        x_center=None,
        stepsize_x=0,
        height=0,
        y_center=None,
        stepsize_y=0,
        dwell=0,
        xrf_on=True,
        ptycho_on=False,
        preamp_on=False,
        position_stream=False)->Annotated[str, "Acquire an image at the given location using Bluesky RunEngine"]:
        
        """Bluesky RunEngine that drives the x- and y- motors to the given location and yields a 2D image.
        
        Parameters
        ----------
        samplename: str
            The name of the sample
        width: float
            The width of the scan
        x_center: float
            The center of the scan in the x direction
        stepsize_x: float
            The step size in the x direction
        height: float
            The height of the scan
        y_center: float
            The center of the scan in the y direction
        stepsize_y: float
            The step size in the y direction
        dwell: float
            The dwell time in the scan
        xrf_on: bool
            Whether to collect XRF data
        ptycho_on: bool
            Whether to collect Ptycho data
        preamp_on: bool
            Whether to collect Preamp data
        position_stream: bool
            Whether to collect position stream data

        Returns
        -------
        str
            The path of the acquired image saved in hard drive.
        """
        
        try:
            logger.info(f"Acquiring image of size {width}x{height} at location {x_center},{y_center}.")
            self.savedata.update_next_file_name()
            self.RE(self.scanplan(
                samplename=samplename,
                width=width,
                x_center=x_center,
                stepsize_x=stepsize_x,
                height=height,
                y_center=y_center,
                stepsize_y=stepsize_y,
                dwell=dwell,
                xrf_on=xrf_on,
                ptycho_on=ptycho_on,
                preamp_on=preamp_on,
                position_stream=position_stream,
            ))
            
            ##TODO: process the .h5 files to get the image
            ##TODO: save the image to the temp directory
            ##TODO: return the path of the image

            return self.savedata.next_file_name
        except Exception as e:
            logger.error(f"Error acquiring image: {e}")
            raise e
        


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
