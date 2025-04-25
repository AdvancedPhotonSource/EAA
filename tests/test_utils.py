import random
import os
import datetime
import logging
from typing import Union

import numpy as np
import pytest


class BaseTester:
    def setup_method(
        self,
        name="",
        generate_data=False,
        generate_gold=False,
        debug=False,
        action=None,
        pytestconfig=None,
    ):
        """
        A Pytest hook that sets instance attributes before running each test method. 
        If the script is executed with `python`, this method will not run automatically
        before calling a test method. Therefore, it can be used to set instance attributes
        for all methods in a code snippet if that snippet is intended to be executed
        with `python`.

        Parameters
        ----------
        name : str, optional
            The name of the tester.
        generate_data : bool
            Whether to generate test data. 
        generate_gold : bool
            Whether to generate gold data. 
        save_timing : bool
            Whether to save timing results.
        debug : bool, optional
            Switches debug mode.
        """
        logging.basicConfig(level=logging.INFO)
        
        self.name = name
        
        self.generate_data = generate_data
        self.generate_gold = generate_gold
        # self.save_timing = save_timing
        self.debug = debug
        
        if pytestconfig is not None:
            self.high_tol = pytestconfig.getoption("high_tol")
            self.action = pytestconfig.getoption("action")
            self.save_timing = pytestconfig.getoption("save_timing")
        else:
            self.high_tol = False
            self.action = action
            self.save_timing = False
            
        self.atol = 1e-3
        self.rtol = 1e-2 if self.high_tol else 1e-4
    
    @pytest.fixture(autouse=True)
    def inject_config(self, pytestconfig):
        self.pytestconfig = pytestconfig
        self.setup_method(
            name="",
            generate_data=False,
            generate_gold=False,
            debug=False,
            action=None,
            pytestconfig=pytestconfig,
        )
    
    @staticmethod
    def get_ci_data_dir():
        try:
            dir = os.environ['EAA_CI_DATA_DIR']
        except KeyError:
            raise KeyError('EAA_CI_DATA_DIR not set. Please set it to the path to the data folder.')
        return dir

    def get_ci_input_data_dir(self):
        return os.path.join(self.get_ci_data_dir(), 'data')

    def get_ci_gold_data_dir(self):
        return os.path.join(self.get_ci_data_dir(), 'gold_data')
    
    def save_gold_data(self, name, data):
        fname = os.path.join(self.get_ci_gold_data_dir(), name, 'recon.npy')
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        np.save(fname, data)
    
    def load_gold_data(self, name):
        fname = os.path.join(self.get_ci_gold_data_dir(), name, 'recon.npy')
        return np.load(fname)
    
    def run_comparison(self, name, test_data, atol=None, rtol=None):
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        gold_data = self.load_gold_data(name)
        compare_data(test_data, gold_data, atol=atol, rtol=rtol, name=name)
        return
    
    def get_default_input_data_file_paths(self, name):
        dp = os.path.join(self.get_ci_input_data_dir(), name, 'ptychodus_dp.hdf5')
        para = os.path.join(self.get_ci_input_data_dir(), name, 'ptychodus_para.hdf5')
        return dp, para

    @staticmethod
    def wrap_comparison_tester(name="", run_comparison=True):
        """
        A decorator factory that wraps a test method to generate or compare data.

        Parameters
        ----------
        name : str, optional
            The name of the test.
        """
        def decorator(test_method):
            def wrapper(self: BaseTester):
                res = test_method(self)
                if self.debug and not self.generate_gold:
                    self.plot_result(res)
                if self.generate_gold:
                    self.save_gold_data(name, res)
                if not self.generate_gold and run_comparison:
                    self.run_comparison(name, res, atol=self.atol, rtol=self.rtol)
            return wrapper
        return decorator
    
    def plot_result(self, res):
        plot_image(res)


def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M')


def compare_data(test_data, gold_data, atol=1e-7, rtol=1e-7, name=""):
    if not np.allclose(gold_data.shape, test_data.shape):
        print('{} FAILED [SHAPE MISMATCH]'.format(name))
        print('  Gold shape: {}'.format(gold_data.shape))
        print('  Test shape: {}'.format(test_data.shape))
        raise AssertionError
    if not np.allclose(gold_data, test_data, atol=atol, rtol=rtol):
        print('{} FAILED [MISMATCH]'.format(name))
        abs_diff = np.abs(gold_data - test_data)
        loc_max_diff = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        loc_max_diff = [a.item() for a in loc_max_diff]
        print('  Mean abs diff: {}'.format(abs_diff.mean()))
        print('  Location of max diff: {}'.format(loc_max_diff))
        print('  Max abs diff: {}'.format(abs_diff[*loc_max_diff].item()))
        print('  Value at max abs diff (test): {}'.format(test_data[*loc_max_diff].item()))
        print('  Value at max abs diff (gold): {}'.format(gold_data[*loc_max_diff].item()))
        raise AssertionError
    print('{} PASSED'.format(name))
    
    
def plot_image(img):
    import matplotlib.pyplot as plt
    if img.ndim == 3:
        img = img[0]
    plt.imshow(img)
    plt.show()


def plot_complex_image(img):
    import matplotlib.pyplot as plt
    if img.ndim == 3:
        img = img[0]
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.abs(img))
    ax[0].set_title('magnitude')
    ax[1].imshow(np.angle(img))
    ax[1].set_title('phase')
    plt.show()

