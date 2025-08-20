
import os
import argparse

import tifffile
import numpy as np

from eaa.tools.imaging.acquisition import SimulatedAcquireImage
from eaa.tools.imaging.registration import ImageRegistration
import test_utils as tutils


class TestImageRegistrationTool(tutils.BaseTester):

    def test_image_registration(self):
        np.random.seed(123)
        
        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )
        
        acquisition_tool = SimulatedAcquireImage(whole_image, return_message=False)
        registration_tool = ImageRegistration(acquisition_tool)
        
        acquisition_tool.acquire_image(
            loc_y=120, loc_x=120, size_y=128, size_x=128
        )
        acquisition_tool.acquire_image(
            loc_y=100, loc_x=100, size_y=128, size_x=128
        )
        
        offset = registration_tool.get_offset_of_latest_image(register_with="previous")
        
        if self.debug:
            print("Offset: ", offset)
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(acquisition_tool.image_k)
            axs[0].set_title(f"Image {acquisition_tool.counter_acquire_image}")
            axs[1].imshow(acquisition_tool.image_km1)
            axs[1].set_title(f"Image {acquisition_tool.counter_acquire_image - 1}")
            plt.show()
        
        assert np.allclose(offset, [20, 20])
        return
    
    def test_image_registration_diff_size(self):
        np.random.seed(123)
        
        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )
        
        acquisition_tool = SimulatedAcquireImage(whole_image, return_message=False)
        registration_tool = ImageRegistration(acquisition_tool, image_coordinates_origin="top_left")
        
        acquisition_tool.acquire_image(
            loc_y=120, loc_x=120, size_y=128, size_x=128
        )
        acquisition_tool.acquire_image(
            loc_y=100, loc_x=100, size_y=150, size_x=150
        )
        
        offset = registration_tool.get_offset_of_latest_image(register_with="previous")
        
        if self.debug:
            print("Offset: ", offset)
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(acquisition_tool.image_k)
            axs[0].set_title(f"Image {acquisition_tool.counter_acquire_image}")
            axs[1].imshow(acquisition_tool.image_km1)
            axs[1].set_title(f"Image {acquisition_tool.counter_acquire_image - 1}")
            plt.show()
        
        assert np.allclose(offset, [20, 20])
        return
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestImageRegistrationTool()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_image_registration()
    tester.test_image_registration_diff_size()
