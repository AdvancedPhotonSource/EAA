
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
    
    def test_simulated_line_scan(self):
        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )
        whole_image = -1 * whole_image.astype(np.float32) + whole_image.max()
        
        tool = SimulatedAcquireImage(whole_image, return_message=False)
        
        fname = tool.scan_line(
            start_y=140,
            end_y=140,
            start_x=408,
            end_x=415,
            scan_step=0.2,
        )
        return fname
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestSimulatedImageAcquisition()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_simulated_image_acquisition()
    tester.test_simulated_line_scan()
        