from typing import Annotated, Tuple, Optional
import logging
import os
import time

from eaa.tools.imaging.acquisition import AcquireImage
from eaa.tools.imaging.aps_mic.util import process_xrfdata, save_xrfdata

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
        preamp1_on: bool = False,
        using_xrf_maps: bool = False,
        xrf_elms: Tuple[str, ...] = ("Cr",),
        allowable_x_range: Optional[Tuple[float, float]] = None,
        allowable_y_range: Optional[Tuple[float, float]] = None,
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
        preamp1_on : bool, optional
            Whether to collect Preamp1 data.
        using_xrf_maps: bool, optional
            Whether to use the XRF-Maps executable to process the data.
        xrf_elms: Tuple[str, ...], optional
            The elements to be detected in the XRF data.
        allowable_x_range: Optional[Tuple[float, float]], optional
            The allowable range of scan center position in the x direction.
        allowable_y_range: Optional[Tuple[float, float]], optional
            The allowable range of scan center position in the y direction.

        Raises
        ------
        ImportError
            If Bluesky control initialization fails.
        """
        try:
            from eaa.tools.imaging.aps_mic.bluesky_init import RE, fly2d, get_control_components
            self.RE = RE
            self.scanplan = fly2d
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
        self.preamp1_on = preamp1_on
        self.using_xrf_maps = using_xrf_maps
        self.xrf_elms = xrf_elms
        self.allowable_x_range = allowable_x_range
        self.allowable_y_range = allowable_y_range
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
            The width of the scan area in microns.
        height: float
            The height of the scan area in microns.
        x_center: float
            The center of the scan area in the x direction in microns.
        y_center: float
            The center of the scan area in the y direction in microns.
        stepsize_x: float
            The scan step size in the x direction, i.e., the distance between
            two adjacent pixels in the x direction in microns.
        stepsize_y: float
            The scan step size in the y direction, i.e., the distance between
            two adjacent pixels in the y direction in microns.

        Returns
        -------
        str
            The path of the acquired image saved in hard drive.
        """
        try:
            if self.allowable_x_range:
                if x_center < self.allowable_x_range[0] or x_center > self.allowable_x_range[1]:
                    raise ValueError(
                        f"The scan center position in the x direction {x_center} um is out "
                        f"of the allowable range {self.allowable_x_range} um."
                    )
            if self.allowable_y_range:
                if y_center < self.allowable_y_range[0] or y_center > self.allowable_y_range[1]:
                    raise ValueError(
                        f"The scan center position in the y direction {y_center} um is out "
                        f"of the allowable range {self.allowable_y_range} um."
                    )
            
            logger.info(f"Acquiring image of size {width} um x {height} um at location {x_center} um, {y_center} um.")
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
                preamp1_on=self.preamp1_on,
            ))

            mda_path = self.savedata.full_path_name.get()
            mda_dir = mda_path.replace("data1", "mnt/micdata1")
            parent_dir = os.path.dirname(os.path.dirname(mda_dir))
            png_output_dir = os.path.join(parent_dir, "png_output")
            current_mda_file = self.savedata.next_file_name

            logger.info(f"About to process the data... {current_mda_file}")
            if self.using_xrf_maps:
                logger.info("Calling the XRF-Maps executable to process the data...")
                process_code = process_xrfdata(parent_dir, current_mda_file)
            else:
                logger.info("Assuming the data is already processed, wait till the .h5 file exists...")
                img_h5_path = os.path.join(
                    os.path.join(parent_dir, "img.dat"),
                    f"{current_mda_file}.h50")
                logger.info(f"The expected .h5 file path is {img_h5_path}")
                
                time_diff = 0
                timenow = time.time()
                while any([time_diff < 30, not os.path.exists(img_h5_path)]):
                    time.sleep(1)
                    if os.path.exists(img_h5_path):
                        time_diff = os.path.getmtime(img_h5_path) - timenow
                        timenow = time.time()
                        logger.info(f"The .h5 file {img_h5_path} exists")
                        logger.info("watch file and wait until the file doesn't change for 30 seconds to process.")
                    else:
                        logger.info(f"The .h5 file {img_h5_path} does not exist")
                        logger.info("wait for 30 seconds to process.")
                        time.sleep(30)

                process_code = 1

            if process_code: 
                logger.info(f"Fitting {current_mda_file} completed successfully.")
                if not os.path.exists(png_output_dir):
                    os.makedirs(png_output_dir)

                img_h5_path = os.path.join(
                    os.path.join(parent_dir, "img.dat"),
                    f"{current_mda_file}.h50")

                img_path = save_xrfdata(img_h5_path, png_output_dir, elms=self.xrf_elms)
                if img_path:
                    return img_path
                else:
                    logger.error(f"Failed to save images for {current_mda_file}")
                    return None
            logger.error(f"Failed to process {current_mda_file}")
            return None
        
        except Exception as e:
            logger.error(f"Error acquiring image: {e}")
            raise e
