from typing import Optional
import numpy as np
import torch
import os
import time
import logging
from math import inf

logger = logging.getLogger(__name__)


def to_tensor(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        # If CUDA is available, convert array to tensor using torch.tensor, which honors the set default device.
        # Otherwise, use from_numpy which creates a reference to the data in memory instead of creating a copy.
        if torch.cuda.is_available():
            return torch.tensor(x)
        else:
            try:
                return torch.from_numpy(x)
            except TypeError:
                return torch.tensor(x)
    else:
        return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return np.asarray(x)
    else:
        return x


def wait_for_file(
    file_path: str,
    duration: int = 30,
    timeout: Optional[int] = inf
) -> bool:
    """Wait for a file to exist and stay unchanged for a given timeout.
    The function only returns when the given file 
    (1) exists, and
    (2) its modification time does not change for `duration` seconds.
    
    If `timeout` is given, the function returns False if the above conditions
    are not met within `timeout` seconds.
    
    Parameters
    ----------
    file_path : str
        The path to the file to wait for.
    duration : int, optional
        The duration that the file is expected to stay unchanged to be
        considered as fully written.
    timeout : int, optional
        The timeout to wait for the file to exist and stay unchanged,
        given in seconds.
        
    Returns
    -------
    bool
        True if the file exists and stays unchanged for given a duration 
        within the given timeout, 
        False otherwise.
    """
    time_diff = 0
    time_mod = 0
    while any([time_diff < duration, not os.path.exists(file_path)]):
        if timeout is not None:
            if time_diff > timeout:
                return False
        time.sleep(1)
        if os.path.exists(file_path):
            if os.path.getmtime(file_path) != time_mod:
                time_mod = os.path.getmtime(file_path)
            time_diff = time.time() - time_mod
            logger.info(f"File {file_path} exists.")
            logger.info(f"Watching file and wait until the file doesn't change for {duration} seconds to process.")
        else:
            logger.info(f"File {file_path} does not exist.")
            logger.info(f"Waiting for {duration} seconds to process.")
            time.sleep(duration)
    return True
