"""
This module contains the functions for processing data collected by the XRF detector.
"""

import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Optional, Tuple
from mic_vis.s2idd.mda import get_roi_from_mda
import eaa.maths

logger = logging.getLogger(__name__)


def validate_position_in_range(
    center: Optional[float],
    allowable_range: Optional[Tuple[float, float]],
    axis_label: str,
) -> None:
    """Validate that a center position lies within an allowable range."""
    if allowable_range is None:
        return

    if len(allowable_range) != 2:
        raise ValueError(
            f"The allowable range for the {axis_label} direction must contain exactly two values."
        )

    lower, upper = allowable_range
    if lower > upper:
        raise ValueError(
            f"The allowable range for the {axis_label} direction "
            f"({allowable_range}) has the lower bound greater than the upper bound."
        )

    if center is None:
        raise ValueError(
            f"The scan center position in the {axis_label} direction must be provided "
            "when an allowable range is set."
        )

    if not lower <= center <= upper:
        raise ValueError(
            f"The scan center position in the {axis_label} direction {center} um is out "
            f"of the allowable range {allowable_range} um."
        )


def run_xrfmaps_exe(exe_path, args=None):
    """
    Runs an executable file.

    Parameters
    ----------
    exe_path : str
        The path to the executable file.
    args : list, optional
        A list of arguments to pass to the executable. Defaults to None.

    Returns
    -------
    subprocess.CompletedProcess
        An object containing information about the completed process.
    """
    command = [exe_path]
    if args:
        command.extend(args)
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error executing {exe_path}: {e}")
        return None


def process_xrfdata(
    parent_dir,
    scan_num_mda,
    det_range="0:0",
    quantify_with="maps_standardinfo.txt",
    fitting_type="roi, nnls",
    num_iter=20000,
    exe_path="/mnt/micdata1/XRF-Maps/bin/xrf_maps.exe"
):
    """
    Runs an executable file.

    Parameters
    ----------
    exe_path : str
        The path to the executable file.
    parent_dir : str
        The path to the parent directory.
    scan_num_mda : str
        The name of the mda file.
    det_range : str
        The range of the detector.
    quantify_with : str
        The name of the quantify with.
    fitting_type : str
        The type of fitting.
    num_iter : int
        The number of iterations.

    Returns
    -------
    subprocess.CompletedProcess
        An object containing information about the completed process.
    """

    mda_dir = os.path.join(parent_dir, "mda")
    mda_files = [f for f in os.listdir(mda_dir) if f.endswith(".mda")]
    if scan_num_mda in mda_files:
        args = [
            "--dir",
            parent_dir,
            "--files",
            scan_num_mda,
            "--detector-range",
            det_range,
            "--export-csv",
            "",
            "--quantify-with",
            quantify_with,
            "--fit",
            fitting_type,
            "--optimizer-num-iter",
            str(num_iter),
            "--generate-avg-h5",
            "",
        ]

        print(f"--------------- fitting {scan_num_mda} ---------------")
        result = run_xrfmaps_exe(exe_path, args)
        if result:
            print(f"--------------- fitting {scan_num_mda} completed ---------------")
            return result.returncode
        else:
            return None
    else:
        print(f"MDA file {scan_num_mda} not found in {mda_dir}")
        return None


def plot_xrf_line_scan(x, y, val_gauss, fwhm, scan_name, roi_num):
    """
    Plot the XRF line scan data.
    """
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.plot(x, y, label="data")
    ax.plot(x, val_gauss, linestyle="--", color="red", label="fit")
    ax.text(
        0.05,
        0.95, 
        f"FWHM = {fwhm:.2f}", 
        transform=ax.transAxes, 
        verticalalignment="top", 
        horizontalalignment="left"
    )

    ax.legend()
    ax.set_xlabel("X-axis Position")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{scan_name}-{roi_num}")
    ax.grid(True)
    plt.tight_layout()
    return fig


def save_xrf_line_scan(
    mda_path: str, 
    output_dir: str, 
    roi_num: int,
    y_threshold: float = 0.0,
    return_line_array: bool = False
) -> str | None:

    """
    Save the XRF line scan data in png format.

    Parameters
    ----------
    mda_path : str
        The path to the MDA file.
    output_dir : str
        The path to the output directory.
    roi_num : int
        The ROI number of the elements of interest.
    y_threshold : float
        The threshold for the Gaussian fit.
    return_line_array : bool
        If True, the line array will be returned.

    Returns
    -------
    str | None
        The path to the saved image.
    """

    try:
        roi_data, position_data = get_roi_from_mda(mda_path, roi_num)
    except Exception as e:
        logger.error(f"Failed to get ROI or position data from MDA file {mda_path}: {e}")
        return None

    if any(data is None for data in [roi_data, position_data]):
        logger.error(f"Failed to get ROI or position data from MDA file {mda_path}")
        return None
    else:
        order = np.argsort(position_data)
        x = position_data[order]
        y = roi_data[order]

        # Fit a Gaussian to the data
        a, mu, sigma, c = eaa.maths.fit_gaussian_1d(x, y, y_threshold=y_threshold)
        val_gauss = eaa.maths.gaussian_1d(x, a, mu, sigma, c)
        fwhm = 2.35 * sigma

        # Plot the data and the fit
        scan_name = os.path.basename(mda_path)
        fig = plot_xrf_line_scan(x, y, val_gauss, fwhm, scan_name, roi_num)
        sc = scan_name.replace(".mda", "")
        fname = f"{output_dir}/{sc}_ROI{roi_num}.png"
        fig.savefig(fname)
        plt.close(fig)
        logger.info(f"Image saved to {fname}")
        if return_line_array:
            return fname, [x, y, val_gauss, fwhm]
        else:
            return fname
