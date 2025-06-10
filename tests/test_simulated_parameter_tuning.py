
import os
import argparse

import tifffile
import numpy as np

from eaa.tools.imaging.acquisition import SimulatedAcquireImage
from eaa.tools.imaging.param_tuning import SimulatedTuneOpticsParameters

import test_utils as tutils


class TestSimulatedParameterTuning(tutils.BaseTester):

    @tutils.BaseTester.wrap_comparison_tester(name='test_simulated_parameter_tuning')
    def test_simulated_parameter_setting(self):
        np.random.seed(123)
        
        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )
        
        acquisition_tool = SimulatedAcquireImage(whole_image, return_message=False)
        tuning_tool = SimulatedTuneOpticsParameters(
            acquisition_tool, 
            true_parameters=[1.0, 2.0, 3.0],
            parameter_ranges=[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]],
            blur_factor=5,
            drift_factor=200,
        )
        
        loc = (100, 100)
        size = (128, 128)
        
        tuning_tool.set_parameters([1.0, 1.0, 0.0])
        img = acquisition_tool.acquire_image(*loc, *size)
        
        if self.debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(img)
            plt.show()
        
        tuning_tool.set_parameters([1.0, 2.0, 2.5])
        img = acquisition_tool.acquire_image(*loc, *size)
        
        if self.debug:
            plt.figure()
            plt.imshow(img)
            plt.show()
        
        return img
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestSimulatedParameterTuning()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_simulated_parameter_setting()

        