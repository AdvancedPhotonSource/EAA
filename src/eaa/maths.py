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
) -> tuple[float, float, float, float, float, float, float]:
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
    tuple[float, float, float, float, float, float, float]
        The amplitude, mean, standard deviation, constant offset, and normalized
        residual of the Gaussian fit, followed by x-range endpoints
        ``(x_min, x_max)`` of input data. The normalized residual is defined as
        ``mean(((y_data - y_fit) / a) ** 2)`` where ``a`` is the fitted amplitude.
    """
    x_data = np.array(x, dtype=float)
    y_data = np.array(y, dtype=float)
    x_min = float(np.min(x_data))
    x_max_input = float(np.max(x_data))
    y_max, y_min = np.max(y_data), np.min(y_data)
    x_max = x_data[np.argmax(y_data)]
    offset = x_max
    x = x_data - offset
    x_max = 0
    mask = y_data >= y_min + y_threshold * (y_max - y_min)
    a_guess = y_max - y_min
    mu_guess = x_max
    x_above_thresh = x[y_data > y_min + a_guess * 0.2]
    if len(x_above_thresh) >= 3:
        sigma_guess = (x_above_thresh.max() - x_above_thresh.min()) / 2
    else:
        sigma_guess = (x.max() - x.min()) / 2
    c_guess = y_min
    p0 = [a_guess, mu_guess, sigma_guess, c_guess]
    try:
        popt, _ = scipy.optimize.curve_fit(gaussian_1d, x[mask], y_data[mask], p0=p0)
    except RuntimeError:
        logger.error("Failed to fit Gaussian to data. Returning NaN values.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    y_fit = gaussian_1d(x, *popt)
    amplitude = float(popt[0])
    if np.isclose(amplitude, 0.0):
        normalized_residual = np.nan
    else:
        normalized_residual = float(np.mean(((y_data - y_fit) / amplitude) ** 2))
    popt[1] += offset
    return (
        float(popt[0]),
        float(popt[1]),
        float(popt[2]),
        float(popt[3]),
        normalized_residual,
        x_min,
        x_max_input,
    )
