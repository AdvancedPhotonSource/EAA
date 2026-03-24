import logging

import numpy as np
import scipy.ndimage
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
    """Fit a 1D Gaussian to 1D data.

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
    finite_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[finite_mask]
    y_data = y_data[finite_mask]
    if x_data.size < 5:
        logger.error("Too few finite data points for Gaussian fitting. Returning NaN values.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    order = np.argsort(x_data)
    x_data = x_data[order]
    y_data = y_data[order]
    x_min = float(np.min(x_data))
    x_max_input = float(np.max(x_data))
    x_span = x_max_input - x_min
    if not np.isfinite(x_span) or x_span <= 0:
        logger.error("Invalid x range for Gaussian fitting. Returning NaN values.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    y_max, y_min = np.max(y_data), np.min(y_data)
    y_range = float(y_max - y_min)
    if not np.isfinite(y_range) or np.isclose(y_range, 0.0):
        logger.error("Input data are too flat for Gaussian fitting. Returning NaN values.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    smooth_sigma = max(1.0, x_data.size * 0.02)
    y_smooth = scipy.ndimage.gaussian_filter1d(y_data, sigma=smooth_sigma, mode="nearest")
    y_smooth_max = float(np.max(y_smooth))
    y_smooth_min = float(np.min(y_smooth))
    y_smooth_range = y_smooth_max - y_smooth_min
    if not np.isfinite(y_smooth_range) or np.isclose(y_smooth_range, 0.0):
        logger.error("Smoothed data are too flat for Gaussian fitting. Returning NaN values.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    x_peak = float(x_data[np.argmax(y_smooth)])
    offset = x_peak
    x = x_data - offset
    fit_threshold = y_smooth_min + y_threshold * y_smooth_range
    mask = y_smooth >= fit_threshold
    if int(np.count_nonzero(mask)) < 5:
        mask = np.ones_like(y_data, dtype=bool)

    positive_weight = np.clip(y_smooth - y_smooth_min, a_min=0.0, a_max=None)
    if np.sum(positive_weight) > 0:
        mu_guess = float(np.sum(x * positive_weight) / np.sum(positive_weight))
    else:
        mu_guess = 0.0

    width_mask = y_smooth >= (y_smooth_min + 0.5 * y_smooth_range)
    x_above_half = x[width_mask]
    if x_above_half.size >= 2:
        sigma_guess = float((x_above_half.max() - x_above_half.min()) / 2.355)
    else:
        sigma_guess = x_span / 6.0
    sigma_guess = float(np.clip(sigma_guess, x_span / 100.0, x_span))

    a_guess = max(y_smooth_range, np.finfo(float).eps)
    c_guess = float(np.median(y_data[~mask])) if np.any(~mask) else float(y_smooth_min)
    p0 = [a_guess, mu_guess, sigma_guess, c_guess]
    lower_bounds = [0.0, float(np.min(x)), max(x_span / 1000.0, 1e-12), float(y_min - y_range)]
    upper_bounds = [float(2 * y_range + abs(y_max)), float(np.max(x)), float(2 * x_span), float(y_max + y_range)]

    def run_fit(current_mask: np.ndarray, current_p0: list[float]) -> np.ndarray | None:
        if int(np.count_nonzero(current_mask)) < 5:
            return None
        try:
            popt, _ = scipy.optimize.curve_fit(
                gaussian_1d,
                x[current_mask],
                y_data[current_mask],
                p0=current_p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )
            return popt
        except (RuntimeError, ValueError):
            return None

    popt = run_fit(mask, p0)
    if popt is None:
        y_smooth_retry = scipy.ndimage.gaussian_filter1d(
            y_data,
            sigma=max(2.0, x_data.size * 0.05),
            mode="nearest",
        )
        retry_weight = np.clip(y_smooth_retry - np.min(y_smooth_retry), a_min=0.0, a_max=None)
        if np.sum(retry_weight) > 0:
            mu_retry = float(np.sum(x * retry_weight) / np.sum(retry_weight))
        else:
            mu_retry = 0.0
        retry_mask = y_smooth_retry >= (
            np.min(y_smooth_retry) + max(0.1, y_threshold) * (np.max(y_smooth_retry) - np.min(y_smooth_retry))
        )
        retry_p0 = [a_guess, mu_retry, sigma_guess, c_guess]
        popt = run_fit(retry_mask, retry_p0)

    if popt is None:
        logger.error("Failed to fit Gaussian to data. Returning NaN values.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input

    y_fit = gaussian_1d(x, *popt)
    amplitude = float(popt[0])
    sigma = float(popt[2])
    mu = float(popt[1])
    if sigma <= 0 or not (float(np.min(x)) <= mu <= float(np.max(x))):
        logger.error("Gaussian fit parameters are invalid. Returning NaN values.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, x_min, x_max_input
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
