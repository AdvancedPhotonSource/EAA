"""
This module contains the functions for processing data collected by the XRF detector.
"""

import subprocess
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


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


def load_h5(img_h5_path, fit_type=["NNLS", "ROI"], fsizelim=1e3) -> dict:
    """
    Load the XRF data from the h5 file.

    Parameters
    ----------
    img_h5_path : str
        The path to the h5 file.
    fit_type : list
        The type of fitting.
    fsizelim : float
        The size limit of the h5 file, only load
        the h5 file larger than this size.

    Returns
    -------
    dict
        The data from the h5 file.
    """
    data = {}
    fsize = os.path.getsize(img_h5_path)
    if fsize > fsizelim:
        with h5py.File(img_h5_path, "r") as f:
            data.update({"scan": os.path.basename(img_h5_path)})
            data.update({"x_axis": f["MAPS/Scan/x_axis"][:]})
            data.update({"y_axis": f["MAPS/Scan/y_axis"][:]})
            for t in fit_type:
                d = f[f"MAPS/XRF_Analyzed/{t}/Counts_Per_Sec"][:]
                d_ch = f[f"MAPS/XRF_Analyzed/{t}/Channel_Names"][:].astype(str).tolist()
                scaler_names = f["MAPS/Scalers/Names"][:].astype(str).tolist()
                scaler_values = f["MAPS/Scalers/Values"][:]

                data.update({f"{t}_arr": d})
                data.update({f"{t}_ch": d_ch})
                data.update({f"{t}_scaler_names": scaler_names})
                data.update({f"{t}_scaler_values": scaler_values})

        return data
    else:
        print(f"The XRF h5 file {img_h5_path} not found")
        return None


def plot_xrfdata(plotarr, xaxis, yaxis, scan_name, elm_name, cmap, vmax, vmin):
    """
    Plot the XRF data.

    Parameters
    ----------
    plotarr : numpy.ndarray
        The array to plot.
    xaxis : numpy.ndarray
        The x-axis data.
    yaxis : numpy.ndarray
        The y-axis data.
    scan_name : str
        The name of the scan.
    elm_name : str
        The name of the element.
    cmap : str
        The colormap to use.
    vmax : float
        The maximum value of the colorbar.
    vmin : float
        The minimum value of the colorbar.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    img = ax.imshow(plotarr, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(f"{scan_name} {elm_name}")

    # Show only 5 ticks for both x- and y- axes
    xticks = np.linspace(0, len(xaxis) - 1, 5, dtype=int)
    yticks = np.linspace(0, len(yaxis) - 1, 5, dtype=int)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([np.round(xaxis[i], 2) for i in xticks])
    ax.set_yticklabels([np.round(yaxis[i], 2) for i in yticks])
    ax.tick_params(axis="both", which="major", labelsize=12)
    # cbar = fig.colorbar(img, ax=ax, shrink=0.3)
    # cbar.set_label("cts/sec")
    plt.tight_layout()
    return fig


def save_xrfdata(
    img_h5_path, output_dir, cmap="inferno", elms=None, vmax_th=99, vmin=0
) -> str | None:
    """
    Save the XRF data in png format.

    Parameters
    ----------
    img_h5_path : str
        The path to the h5 file.
    output_dir : str
        The path to the output directory.
    cmap : str
        The colormap to use.
    elms : list
        The elements to plot.
    vmax_th : float
        The threshold for the maximum percentile of the colorbar.
    vmin : float
        The minimum value of the colorbar.

    Returns
    -------
    str | None
        The path to the saved image.
    """
    data = load_h5(img_h5_path)
    if data:
        data_arr = data["ROI_arr"]
        data_ch = data["ROI_ch"]
        xaxis = data["x_axis"]
        yaxis = data["y_axis"]
        if elms:
            plot_elms = elms
        else:
            plot_elms = data_ch

        for e in plot_elms:
            plotarr = data_arr[data_ch.index(e)]
            vmax = np.nanpercentile(plotarr, vmax_th)
            fig = plot_xrfdata(plotarr, xaxis, yaxis, data["scan"], e, cmap, vmax, vmin)
            fname = f"{output_dir}/{data['scan']}_{e}.png"
            fig.savefig(fname)
            plt.close(fig)
            logger.info(f"Image saved to {fname}")
            return fname
    else:
        logger.error(f"The XRF h5 file {img_h5_path} not found")
        return None
