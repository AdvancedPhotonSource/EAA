import numpy as np
import scipy.optimize


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
        The amplitude, mean, and standard deviation of the Gaussian.
    """
    y_max, y_min = np.max(y), np.min(y)
    x_max = x[np.argmax(y)]
    mask = y >= y_min + y_threshold * (y_max - y_min)
    p0 = [y_max - y_min, x_max, np.count_nonzero(mask) / (x[-1] - x[0]) / 2, y_min]
    popt, _ = scipy.optimize.curve_fit(gaussian_1d, x[mask], y[mask], p0=p0)
    return tuple(popt)
