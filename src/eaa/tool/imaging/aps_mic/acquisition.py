from typing import Annotated, Tuple, Optional
from io import BytesIO
import logging
import os
import json

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sciagent.tool.base import ToolReturnType, tool

from eaa.tool.imaging.aps_mic.util import (
    process_xrfdata,
    save_xrf_line_scan,
    validate_position_in_range,
    save_xrfdata
)
from eaa.util import wait_for_file
from eaa.tool.imaging.acquisition import AcquireImage

logger = logging.getLogger(__name__)


class BlueSkyAcquireImage(AcquireImage):

    from bluesky.run_engine import RunEngine
    from typing import Callable
    from mic_common.devices.save_data import SaveDataMic
    from mic_vis.s2idd.xrf_eaa import save_xrfdata
    from ophyd import EpicsMotor
    import bluesky.plan_stubs as bps
    
    name: str = "bluesky_acquire_image"
    RE: RunEngine = None
    savedata: SaveDataMic = None
    scan2d_plan: Callable = None
    scan1d_plan: Callable = None
    samy_motor: EpicsMotor = None
    bps: Callable = bps
    
    def __init__(
        self, 
        sample_name: str = "smp1",
        dwell_imaging: float = 0.05,
        dwell_line_scan: float = 0.2,
        xrf_on: bool = True,
        preamp1_on: bool = False,
        using_xrf_maps: bool = False,
        xrf_elms: Tuple[str, ...] = ("Cr",),
        xrf_roi_num: int = 16,
        allowable_x_range: Optional[Tuple[float, float]] = None,
        allowable_y_range: Optional[Tuple[float, float]] = None,
        allowable_z_range: Optional[Tuple[float, float]] = None,
        plot_image_in_log_scale: bool = False,
        show_colorbar_in_image: bool = False,
        require_approval: bool = False,
        line_scan_return_gaussian_fit: bool = False,
        *args, **kwargs
    ):
        """Image acquisition tool with Bluesky.

        Parameters
        ----------
        sample_name : str, optional
            The name of the sample.
        dwell_imaging : float, optional
            The dwell time in the unit of seconds for imaging.
        dwell_line_scan : float, optional
            The dwell time in the unit of seconds for line scan.
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
        plot_image_in_log_scale: bool, optional
            Whether to plot the image in log scale.
        show_colorbar_in_image: bool, optional
            Whether to show the colorbar in the image.
        line_scan_return_gaussian_fit: bool, optional
            If True, the function returns a stringified JSON object containing the image path
            and the Gaussian fit FWHM.
        
        Raises
        ------
        ImportError
            If Bluesky control initialization fails.
        """
        self.sample_name = sample_name
        self.dwell_imaging = dwell_imaging
        self.dwell_line_scan = dwell_line_scan
        self.xrf_on = xrf_on
        self.preamp1_on = preamp1_on
        self.using_xrf_maps = using_xrf_maps
        self.xrf_elms = xrf_elms
        self.xrf_roi_num = xrf_roi_num
        self.allowable_x_range = allowable_x_range
        self.allowable_y_range = allowable_y_range
        self.allowable_z_range = allowable_z_range
        self.plot_image_in_log_scale = plot_image_in_log_scale
        self.show_colorbar_in_image = show_colorbar_in_image
        self.line_scan_return_gaussian_fit = line_scan_return_gaussian_fit
        super().__init__(*args, require_approval=require_approval, **kwargs)
        
        
    @tool(name="acquire_image", return_type=ToolReturnType.IMAGE_PATH)
    def acquire_image(
        self,
        width: Annotated[float, "The width of the scan area in microns"] = 0,
        height: Annotated[float, "The height of the scan area in microns"] = 0,
        x_center: Annotated[float, "The center of the scan area in the x direction in microns"] = None,
        y_center: Annotated[float, "The center of the scan area in the y direction in microns"] = None,
        stepsize_x: Annotated[float, "The scan step size in the x direction, i.e., the distance between two adjacent pixels in the x direction in microns"] = 0,
        stepsize_y: Annotated[float, "The scan step size in the y direction, i.e., the distance between two adjacent pixels in the y direction in microns"] = 0,
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
        self.update_image_acquisition_call_history(x_center, y_center, width, height, stepsize_x, stepsize_y)
        try:
            validate_position_in_range(x_center, self.allowable_x_range, "x")
            validate_position_in_range(y_center, self.allowable_y_range, "y")
            logger.info(f"Acquiring image of size {width} um x {height} um at location {x_center} um, {y_center} um.")
            
            if self.RE is not None:
                self.RE(self.scan2d_plan(
                    samplename=self.sample_name,
                    width=width,
                    x_center=x_center,
                    stepsize_x=stepsize_x,
                    height=height,
                    y_center=y_center,
                    stepsize_y=stepsize_y,
                    dwell_ms=self.dwell_imaging*1000,
                    xrf_on=self.xrf_on,
                    preamp1_on=self.preamp1_on,
                ))
            else:
                raise ValueError("RunEngine is not initialized.")

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
                
                process_code = wait_for_file(img_h5_path, duration=30)

            if process_code: 
                logger.info(f"Fitting {current_mda_file} completed successfully.")
                if not os.path.exists(png_output_dir):
                    os.makedirs(png_output_dir)

                img_h5_path = os.path.join(
                    os.path.join(parent_dir, "img.dat"),
                    f"{current_mda_file}.h50")

                img_path, img_arr = save_xrfdata(
                    img_h5_path, 
                    png_output_dir, 
                    elms=self.xrf_elms, 
                    return_image_array=True,
                    plot_in_log_scale=self.plot_image_in_log_scale,
                    show_colorbar_in_image=self.show_colorbar_in_image
                )
                wait_for_file(img_path, duration=5)

                self.update_image_buffers(img_arr, psize=stepsize_x)
                if img_path:
                    return img_path
                else:
                    logger.error(f"Failed to save images for {current_mda_file}")
                    return f"Failed to save images for {current_mda_file}"
            logger.error(f"Failed to process {current_mda_file}")
            return f"Failed to process {current_mda_file}"
        
        except Exception as e:
            logger.error(f"Error acquiring image: {e}")
            raise e


    @tool(name="acquire_line_scan", return_type=ToolReturnType.IMAGE_PATH)
    def acquire_line_scan(
        self,
        width: Annotated[float, "The width of the scan area in microns"] = 0,
        x_center: Annotated[float, "The center of the scan area in the x direction in microns"] = None,
        y_center: Annotated[float, "The center of the scan area in the y direction in microns"] = None,
        stepsize_x: Annotated[float, "The scan step size in the x direction, i.e., the distance between two adjacent pixels in the x direction in microns"] = 0,
        view_scan_line_in_image: Annotated[
            bool,
            "If True, show the line scan position on the latest acquired image.",
        ] = False,

    )->Annotated[str, "The path to the plot of the line scan."]:
        """Acquire a horizontal line scan of a given width at a given center position.
        This function returns a plot of the acquired data, and a Gaussian fit of it.
        The FWHM of the Gaussian fit is annotated on the plot.
        
        Parameters
        ----------
        width: float
            The width of the scan area in microns.
        x_center: float
            The center of the scan area in the x direction in microns.
        y_center: float
            The center of the scan area in the y direction in microns.
        stepsize_x: float
            The scan step size in the x direction, i.e., the distance between
            two adjacent pixels in the x direction in microns.
        view_scan_line_in_image : bool, optional
            If True, show the line scan position on the latest acquired image.

        Returns
        -------
        str
            The path of the plot of the line scan saved in hard drive.

        """
        self.set_motor_y(y_center)
        
        start_x = x_center - width/2
        end_x = x_center + width/2
        self.update_line_scan_call_history(start_x=start_x, end_x=end_x, step=stepsize_x, start_y=None, end_y=None)

        try:
            validate_position_in_range(start_x, self.allowable_x_range, "x")
            validate_position_in_range(end_x, self.allowable_x_range, "x")
            logger.info(f"Acquiring line scan of width {width} um at center location of {x_center} um.")

            if self.RE is not None:
                self.RE(self.scan1d_plan(
                    samplename=self.sample_name,
                    width=width,
                    x_center=x_center,
                    stepsize_x=stepsize_x,
                    dwell_ms=self.dwell_line_scan*1000,
                    xrf_on=self.xrf_on,
                    preamp1_on=self.preamp1_on,
                ))
            else:
                raise ValueError("RunEngine is not initialized.")

            mda_path = self.savedata.full_path_name.get()
            mda_dir = mda_path.replace("data1", "mnt/micdata1")
            parent_dir = os.path.dirname(os.path.dirname(mda_dir))
            png_output_dir = os.path.join(parent_dir, "png_output")
            current_mda_file = self.savedata.next_file_name
            mda_file_path = os.path.join(mda_dir, current_mda_file)

            logger.info(f"About to process the data... {current_mda_file}")
            process_code = wait_for_file(mda_file_path, duration=20)

            if process_code:
                if not os.path.exists(png_output_dir):
                    os.makedirs(png_output_dir)
                
                img_path, [_, _, _, fwhm] = save_xrf_line_scan(
                    mda_file_path, png_output_dir, roi_num=self.xrf_roi_num, 
                    return_line_array=True
                )
                wait_for_file(img_path, duration=5)

                if np.isnan(fwhm):
                    logger.warning("Gaussian fit returned NaN for line scan FWHM.")

                # self.update_line_scan_buffers(img_arr, psize=stepsize_x)
                if img_path:
                    if (
                        view_scan_line_in_image
                        and self.image_k is not None
                        and len(self.image_acquisition_call_history) > 0
                    ):
                        image_info = self.image_acquisition_call_history[-1]
                        image_to_plot = self.image_k
                        if self.plot_image_in_log_scale:
                            image_to_plot = np.log10(image_to_plot + 1)
                        x_min = image_info["loc_x"] - image_info["size_x"] / 2
                        x_max = image_info["loc_x"] + image_info["size_x"] / 2
                        y_min = image_info["loc_y"] - image_info["size_y"] / 2
                        y_max = image_info["loc_y"] + image_info["size_y"] / 2
                        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        ax.imshow(
                            image_to_plot,
                            cmap="gray",
                            origin="upper",
                            extent=[x_min, x_max, y_max, y_min],
                        )
                        ax.plot(
                            [start_x, end_x],
                            [y_center, y_center],
                            color="red",
                            linewidth=2,
                        )
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_title("Line scan position")
                        fig.tight_layout()
                        buffer = BytesIO()
                        fig.savefig(buffer, format="png")
                        plt.close(fig)
                        buffer.seek(0)
                        overlay_image = Image.open(buffer).convert("RGB")
                        line_scan_image = Image.open(img_path).convert("RGB")
                        if overlay_image.height != line_scan_image.height:
                            new_width = int(
                                overlay_image.width
                                * line_scan_image.height
                                / overlay_image.height
                            )
                            overlay_image = overlay_image.resize(
                                (new_width, line_scan_image.height)
                            )
                        stitched_image = Image.new(
                            "RGB",
                            (
                                line_scan_image.width + overlay_image.width,
                                max(line_scan_image.height, overlay_image.height),
                            ),
                            "white",
                        )
                        stitched_image.paste(line_scan_image, (0, 0))
                        stitched_image.paste(overlay_image, (line_scan_image.width, 0))
                        stitched_image.save(img_path)
                    if self.line_scan_return_gaussian_fit:
                        return json.dumps({
                            "image_path": img_path,
                            "fwhm": fwhm,
                        })
                    return img_path
                else:
                    logger.error(f"Failed to save images for {current_mda_file}")
                    return f"Failed to save images for {current_mda_file}"
            logger.error(f"Failed to process {current_mda_file}")
            return f"Failed to process {current_mda_file}"
        
        except Exception as e:
            logger.error(f"Error acquiring line scan: {e}")
            raise e

    def set_motor_y(
        self, 
        value: float
    ) -> str:
        """Set the sample y motor position of the imaging system.
        
        Parameters
        ----------
        value: float
            The value to set the sample y motor to.
        """
        if self.RE is None:
            raise ValueError("RunEngine is not set")
        if self.samy_motor is None:
            raise ValueError("samz_motor is not set")
        
        validate_position_in_range(value, self.allowable_y_range, "y")
        self.RE(self.bps.mv(self.samy_motor, value))
        msg = f"Move sample y motor to position: {value}"
        logger.info(msg)
        return msg
