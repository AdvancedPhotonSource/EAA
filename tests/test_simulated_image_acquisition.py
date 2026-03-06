
from scipy.signal.windows import gaussian
import os
import argparse

import tifffile
import numpy as np

from eaa.tool.imaging.acquisition import SimulatedAcquireImage

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

        # center = top-left (100, 100) + half-size (128, 128)
        center = (228, 228)
        size = (256, 256)
        img = tool.acquire_image(*center, *size)
        return img

    # Deliberately skipping result comparison because of non-deterministicness
    def test_advanced_simulated_image_acquisition(self):
        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )

        tool = SimulatedAcquireImage(
            whole_image,
            poisson_noise_scale=100,
            scan_jitter=1.0,
            gaussian_psf_sigma=1.0,
            return_message=False
        )

        center = (228, 228)
        size = (256, 256)
        img = tool.acquire_image(*center, *size)
        if self.debug:
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
        return None
    
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
        
        fname = tool.acquire_line_scan(
            x_center=411.5,
            y_center=140,
            length=7,
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
    tester.test_advanced_simulated_image_acquisition()
    tester.test_simulated_line_scan()
        