import logging

import numpy as np
import scipy.optimize

logger = logging.getLogger(__name__)


def gaussian_1d(x: np.ndarray, a: float, mu: float, sigma: float, c: float = 0) -> np.ndarray:
    """A 1D Gaussian function.

    Parameters
    ----------
    x : np.ndarray
        The x-values of the data.
    a : float
        The amplitude of the Gaussian.
    mu : float
        The mean of the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.
    c : float, optional
        The constant offset of the Gaussian.

    Returns
    -------
    np.ndarray
        The y-values of the Gaussian.
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c


def fit_gaussian_1d(
    x: np.ndarray,
    y: np.ndarray,
    y_threshold: float = 0,
) -> tuple[float, float, float]:
    """Fit a 1D Gaussian to the data after subtracting a linear background.

    Parameters
    ----------
    x : np.ndarray
        The x-values of the data.
    y : np.ndarray
        The y-values of the data.
    y_threshold : float, optional
        Only points whose y values are above y_min + y_threshold * (y_max - y_min)
        are considered for fitting. To disable point selection, 
        set y_threshold to 0.

    Returns
    -------
    tuple[float, float, float]
        The amplitude, mean, standard deviation, and constant offset of the Gaussian.
    """
    y_max, y_min = np.max(y), np.min(y)
    x_max = x[np.argmax(y)]
    offset = x_max
    x = x - offset
    x_max = 0
    mask = y >= y_min + y_threshold * (y_max - y_min)
    a_guess = y_max - y_min
    mu_guess = x_max
    x_above_thresh = x[y > y_min + a_guess * 0.2]
    if len(x_above_thresh) >= 3:
        sigma_guess = (x_above_thresh.max() - x_above_thresh.min()) / 2
    else:
        sigma_guess = (x.max() - x.min()) / 2
    c_guess = y_min
    p0 = [a_guess, mu_guess, sigma_guess, c_guess]
    try:
        popt, _ = scipy.optimize.curve_fit(gaussian_1d, x[mask], y[mask], p0=p0)
        popt[1] += offset
    except RuntimeError:
        logger.error("Failed to fit Gaussian to data. Returning NaN values.")
        return np.nan, np.nan, np.nan, np.nan
    return tuple(popt)

