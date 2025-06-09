
import os
import argparse

import tifffile
import numpy as np

from eaa.tools.imaging.acquisition import SimulatedAcquireImage

import test_utils as tutils


class TestSimulatedImageAcquisition(tutils.BaseTester):

    @tutils.BaseTester.wrap_comparison_tester(name='test_simulated_image_acquisition')
    def test_simulated_image_acquisition(self):
        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )
        
        tool = SimulatedAcquireImage(whole_image, return_message=False)
        
        loc = (100, 100)
        size = (256, 256)
        img = tool.acquire_image(*loc, *size)
        return img
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestSimulatedImageAcquisition()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_simulated_image_acquisition()

        