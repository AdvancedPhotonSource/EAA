from typing import Annotated, Dict, List, Any
import logging
import os

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
        
        # Buffered images:
        # image_0 - the first image
        # image_km1 - the previous image
        # image_k - the current image
        self.image_0: np.ndarray = None
        self.image_km1: np.ndarray = None
        self.image_k: np.ndarray = None
        self.psize_0 = None
        self.psize_km1 = None
        self.psize_k = None
        
        self.image_acquisition_call_history: List[Dict[str, Any]] = []
        self.line_scan_call_history: List[Dict[str, Any]] = []
                
    @property
    def counter_acquire_image(self):
        return len(self.image_acquisition_call_history)
        
    def update_image_acquisition_call_history(
        self, 
        loc_x: float, 
        loc_y: float, 
        size_x: float, 
        size_y: float,
        psize_x: float,
        psize_y: float
    ):
        self.image_acquisition_call_history.append({
            "loc_x": loc_x,
            "loc_y": loc_y,
            "size_x": size_x,
            "size_y": size_y,
            "psize_x": psize_x,
            "psize_y": psize_y,
        })
        
    def update_line_scan_call_history(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        step: float
    ):
        self.line_scan_call_history.append({
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "step": step
        })
    

    def update_real_time_view(self, image: np.ndarray):
        if self.rt_fig is None:
            self.rt_fig, ax = plt.subplots(1, 1, squeeze=True)
        else:
            ax = self.rt_fig.get_axes()[0]
        ax.clear()
        ax.imshow(image)
        plt.draw()
        plt.pause(0.001)  # Small pause to allow GUI to update
        
    def update_image_buffers(self, new_image: np.ndarray, psize: float = 1):
        """Update the image buffers.
        
        Parameters
        ----------
        new_image : np.ndarray
            The new image.
        psize : float, optional
            The pixel size (or scan step) of the new image.
        """
        if self.counter_acquire_image == 0:
            self.image_0 = new_image
            self.psize_0 = psize
        self.image_km1 = self.image_k
        self.psize_km1 = self.psize_k
        self.image_k = new_image
        self.psize_k = psize

    def acquire_image(self, *args, **kwargs):
        raise NotImplementedError


class SimulatedAcquireImage(AcquireImage):
    
    name: str = "simulated_acquire_image"
    
    def __init__(
        self, 
        whole_image: np.ndarray, 
        return_message: bool = True,
        add_axis_ticks: bool = False,
        n_ticks: int = 10,
        add_grid_lines: bool = False,
        invert_yaxis: bool = False,
        line_scan_gaussian_fit_y_threshold: float = 0,
        add_line_scan_candidates_to_image: bool = False,
        plot_image_in_log_scale: bool = False,
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
        add_grid_lines : bool, optional
            If True, the tool adds grid lines to the image.
        invert_yaxis : bool, optional
            If True, the tool inverts the y-axis of the acquired image.
        line_scan_gaussian_fit_y_threshold : float, optional
            The threshold for the Gaussian fit of the line scan. Only points whose
            y values are above y_min + y_threshold * (y_max - y_min) are considered
            for fitting. To disable point selection, set y_threshold to 0.
        add_line_scan_candidates_to_image : bool, optional
            If True, the tool adds line scan candidates to the image.
        plot_image_in_log_scale : bool, optional
            If True, 2D images are plotted in log scale.
        """
        self.whole_image = whole_image
        self.interpolator = None
        self.blur = None
        self.offset = np.array([0, 0])
        self.line_scan_gaussian_fit_y_threshold = line_scan_gaussian_fit_y_threshold
        
        self.return_message = return_message
        self.add_axis_ticks = add_axis_ticks
        self.n_ticks = n_ticks
        self.add_grid_lines = add_grid_lines
        self.invert_yaxis = invert_yaxis
        self.add_line_scan_candidates_to_image = add_line_scan_candidates_to_image
        self.plot_image_in_log_scale = plot_image_in_log_scale
        
        self.line_scan_candidates: Dict[int, list[int]] = {}

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
        
    def add_line_scan_candidates(
        self, 
        fig: plt.Figure, 
        length: float = 30,
        gap: float = 5,
        spacing: float = 30,
        horizontal: bool = True,
    ):
        """Add markers indicating line scan paths that can be chosen from
        to a figure.
        
        Parameters
        ----------
        fig : plt.Figure
            The figure to add the markers to.
        ny, nx : int
            The number of markers to add in the y and x directions.
        length : float
            The length of the markers.
        gap : float
            The gap between the ends of the markers.
        spacing : float
            The parallel spacing between the markers.
        horizontal : bool, optional
            If True, the markers are added horizontally. If False, the markers
            are added vertically.
        """
        self.line_scan_candidates = {}
        
        ax = fig.get_axes()[0]
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ylim_sorted = sorted(ylim)
        if horizontal:
            start_xs = np.arange(xlim[0], xlim[1], length + gap)
            end_xs = start_xs + length
            start_ys = np.arange(ylim_sorted[0], ylim_sorted[1], spacing)
            end_ys = start_ys
        else:
            start_ys = np.arange(ylim_sorted[0], ylim_sorted[1], length + gap)
            end_ys = start_ys + length
            start_xs = np.arange(xlim[0], xlim[1], spacing)
            end_xs = start_xs
        start_xs_all, start_ys_all = np.meshgrid(start_xs, start_ys, indexing="ij")
        end_xs_all, end_ys_all = np.meshgrid(end_xs, end_ys, indexing="ij")
        start_xs_all = start_xs_all.flatten()
        start_ys_all = start_ys_all.flatten()
        end_xs_all = end_xs_all.flatten()
        end_ys_all = end_ys_all.flatten()
        for i in range(len(start_xs_all)):
            ax.plot([start_xs_all[i], end_xs_all[i]], [start_ys_all[i], end_ys_all[i]], color="red")
            ax.text(
                (start_xs_all[i] + end_xs_all[i]) / 2,
                (start_ys_all[i] + end_ys_all[i]) / 2,
                f"{i}",
                color="red",
                horizontalalignment="center",
                verticalalignment="bottom"
            )
            self.line_scan_candidates[i] = [start_xs_all[i], start_ys_all[i], end_xs_all[i], end_ys_all[i]]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return fig

    def acquire_image(
        self, 
        loc_y: Annotated[float, "The y-coordinate of the top-left corner of the image to acquire."], 
        loc_x: Annotated[float, "The x-coordinate of the top-left corner of the image to acquire."], 
        size_y: Annotated[int, "The height of the image to acquire."], 
        size_x: Annotated[int, "The width of the image to acquire."], 
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
        self.update_image_acquisition_call_history(loc_x, loc_y, size_x, size_y, psize_x=1, psize_y=1)
        
        loc = [loc_y, loc_x]
        size = [size_y, size_x]
        logger.info(f"Acquiring image of size {size} at location {loc}.")
        y = np.arange(loc[0], loc[0] + size[0])
        x = np.arange(loc[1], loc[1] + size[1])
        arr = self.interpolator(y + self.offset[0], x + self.offset[1]).reshape(size)
        
        if self.blur is not None and self.blur > 0:
            arr = ndi.gaussian_filter(arr, self.blur)
        
        if self.show_image_in_real_time:
            self.update_real_time_view(arr)
            
        self.update_image_buffers(arr, psize=1)
            
        if self.return_message:
            filename = f"image_{loc_y}_{loc_x}_{size_y}_{size_x}_{eaa.util.get_timestamp()}.png"
            fig = self.plot_2d_image(
                arr if not self.plot_image_in_log_scale else np.log10(arr + 1),
                add_axis_ticks=self.add_axis_ticks,
                x_ticks=x,
                y_ticks=y,
                n_ticks=self.n_ticks,
                add_grid_lines=self.add_grid_lines,
                invert_yaxis=self.invert_yaxis
            )
            if self.add_line_scan_candidates_to_image:
                fig = self.add_line_scan_candidates(fig)
            self.save_image_to_temp_dir(fig, filename, add_timestamp=False)
            return f".tmp/{filename}"
        else:
            return arr

    def scan_line(
        self,
        start_x: Annotated[float, "The x-coordinate of the starting point of the line scan."],
        start_y: Annotated[float, "The y-coordinate of the starting point of the line scan."],
        end_x: Annotated[float, "The x-coordinate of the ending point of the line scan."],
        end_y: Annotated[float, "The y-coordinate of the ending point of the line scan."],
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
        self.update_line_scan_call_history(start_x, start_y, end_x, end_y, scan_step)
        
        pt_start = np.array([start_y, start_x])
        pt_end = np.array([end_y, end_x])
        d_tot = np.linalg.norm(pt_end - pt_start)
        ds = np.arange(0, d_tot, scan_step)
        pts = pt_start + ds[:, None] * (pt_end - pt_start) / d_tot
        pts = pts + self.offset
        
        arr = self.line_interpolator(pts).reshape(-1)
        
        if self.blur is not None and self.blur > 0:
            arr = ndi.gaussian_filter(arr, self.blur)
            
        # Fit a Gaussian to the line scan
        a, mu, sigma, c = eaa.maths.fit_gaussian_1d(
            ds, arr, y_threshold=self.line_scan_gaussian_fit_y_threshold
        )
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
        
        if not os.path.exists(".tmp"):
            os.makedirs(".tmp")
        fname = os.path.join(
            ".tmp",
            f"line_scan_{start_y}_{end_y}_{start_x}_{end_x}_{scan_step}_{eaa.util.get_timestamp()}.png"
        )
        fig.savefig(fname)
        plt.close(fig)
        return fname

    def scan_line_by_choice(
        self, 
        choice: Annotated[int, "The index of the line scan candidate to use."],
        scan_step: Annotated[float, "The step size of the line scan."] = 1.0,
    ) -> Annotated[str, "The path to the plot of the line scan."]:
        """Conduct a line scan along a chosen path. To use this tool,
        you must call the tool "acquire_image" first, examine the image
        with the candidates, and then call this tool with the index of the
        candidate you want to use.
        
        Parameters
        ----------
        choice : int
            The index of the line scan candidate to use. You should have
            seen an image with the line scan candidates.
        scan_step : float
            The step size of the line scan.

        Returns
        -------
        str
            The path of the plot of the line scan saved in hard drive.
        """
        start_x, start_y, end_x, end_y = self.line_scan_candidates[choice]
        self.update_line_scan_call_history(start_x, start_y, end_x, end_y, scan_step)
        return self.scan_line(start_x, start_y, end_x, end_y, scan_step=scan_step)
