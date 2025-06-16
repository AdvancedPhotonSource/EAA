from typing import Annotated
import logging

from eaa.tools.imaging.acquisition import AcquireImage

logger = logging.getLogger(__name__)


class BlueSkyAcquireImage(AcquireImage):

    from bluesky.run_engine import RunEngine
    from mic_common.devices.save_data import SaveDataMic
    from typing import Callable
    
    name: str = "bluesky_acquire_image"
    RE: RunEngine = None
    scanplan: Callable = None
    savedata: SaveDataMic = None
    
    def __init__(
        self, 
        sample_name: str = "smp1",
        dwell: float = 0,
        xrf_on: bool = True,
        ptycho_on: bool = False,
        preamp_on: bool = False,
        position_stream: bool = False,
        *args, **kwargs
    ):
        """Image acquisition tool with Bluesky.

        Parameters
        ----------
        sample_name : str, optional
            The name of the sample.
        dwell : float, optional
            The dwell time.
        xrf_on : bool, optional
            Whether to collect XRF data.
        ptycho_on : bool, optional
            Whether to collect Ptychography data.
        preamp_on : bool, optional
            Whether to collect Preamp data.
        position_stream : bool, optional
            Whether to collect position stream data.

        Raises
        ------
        ImportError
            If Bluesky control initialization fails.
        """
        try:
            from eaa.tools.imaging.aps_mic.bluesky_init import RE, step2d, get_control_components
            self.RE = RE
            self.scanplan = step2d
            self.savedata = get_control_components("savedata")
        except ImportError:
            raise ImportError(
                "Bluesky control initialization failed. "
                "Please check that the bluesky-mic package is installed "
                "and the motors can only be reached from private subnet computers."
            )
        
        self.sample_name = sample_name
        self.dwell = dwell
        self.xrf_on = xrf_on
        self.ptycho_on = ptycho_on
        self.preamp_on = preamp_on
        self.position_stream = position_stream
        
        super().__init__(*args, **kwargs)
        
    def acquire_image(
        self,
        width: float = 0,
        height: float = 0,
        x_center: float = None,
        y_center: float = None,
        stepsize_x: float = 0,
        stepsize_y: float = 0,
    )->Annotated[str, "The path to the acquired image."]:
        """Acquire an image of a given scan area with the scanning x-ray microscope.
        
        Parameters
        ----------
        width: float
            The width of the scan area.
        height: float
            The height of the scan area.
        x_center: float
            The center of the scan area in the x direction.
        y_center: float
            The center of the scan area in the y direction.
        stepsize_x: float
            The scan step size in the x direction, i.e., the distance between
            two adjacent pixels in the x direction.
        stepsize_y: float
            The scan step size in the y direction, i.e., the distance between
            two adjacent pixels in the y direction.

        Returns
        -------
        str
            The path of the acquired image saved in hard drive.
        """
        try:
            logger.info(f"Acquiring image of size {width}x{height} at location {x_center},{y_center}.")
            self.savedata.update_next_file_name()
            self.RE(self.scanplan(
                samplename=self.sample_name,
                width=width,
                x_center=x_center,
                stepsize_x=stepsize_x,
                height=height,
                y_center=y_center,
                stepsize_y=stepsize_y,
                dwell=self.dwell,
                xrf_on=self.xrf_on,
                ptycho_on=self.ptycho_on,
                preamp_on=self.preamp_on,
                position_stream=self.position_stream,
            ))
            
            ##TODO: add units to lengths and positions in docstring
            ##TODO: process the .h5 files to get the image
            ##TODO: save the image to the temp directory
            ##TODO: return the path of the image

            return self.savedata.next_file_name
        except Exception as e:
            logger.error(f"Error acquiring image: {e}")
            raise e