
import os
import argparse

import tifffile
import numpy as np

from eaa_imaging.tool.imaging.acquisition import SimulatedAcquireImage
from eaa_imaging.tool.imaging.registration import ImageRegistration
import test_utils as tutils


class TestImageRegistrationTool(tutils.BaseTester):
    @staticmethod
    def register_previous_current(acquisition_tool, registration_tool):
        """Register previous and current acquisition buffers."""
        previous_info = acquisition_tool.get_previous_image_info()
        current_info = acquisition_tool.get_current_image_info()
        return registration_tool.register_images(
            image_t=acquisition_tool.image_k,
            image_r=acquisition_tool.image_km1,
            psize_t=current_info["psize"],
            psize_r=previous_info["psize"],
        )

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
        registration_tool = ImageRegistration(registration_method="phase_correlation")
        
        acquisition_tool.acquire_image(
            y_center=184, x_center=184, size_y=128, size_x=128
        )
        acquisition_tool.acquire_image(
            y_center=164, x_center=164, size_y=128, size_x=128
        )

        offset = self.register_previous_current(acquisition_tool, registration_tool)

        if self.debug:
            print("Offset: ", offset)
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(acquisition_tool.image_k)
            axs[0].set_title(f"Image {acquisition_tool.counter_acquire_image}")
            axs[1].imshow(acquisition_tool.image_km1)
            axs[1].set_title(f"Image {acquisition_tool.counter_acquire_image - 1}")
            plt.show()

        assert np.allclose(offset, [-20, -20])
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
        registration_tool = ImageRegistration(
            image_coordinates_origin="top_left",
            registration_method="phase_correlation",
        )
        
        acquisition_tool.acquire_image(
            y_center=184, x_center=184, size_y=128, size_x=128
        )
        acquisition_tool.acquire_image(
            y_center=175, x_center=175, size_y=150, size_x=150
        )

        offset = self.register_previous_current(acquisition_tool, registration_tool)
        
        if self.debug:
            print("Offset: ", offset)
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(acquisition_tool.image_k)
            axs[0].set_title(f"Image {acquisition_tool.counter_acquire_image}")
            axs[1].imshow(acquisition_tool.image_km1)
            axs[1].set_title(f"Image {acquisition_tool.counter_acquire_image - 1}")
            plt.show()
        
        assert np.allclose(offset, [-20, -20])
        return

    def test_image_registration_mutual_information(self):
        np.random.seed(123)

        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )

        acquisition_tool = SimulatedAcquireImage(whole_image, return_message=False)
        registration_tool = ImageRegistration(
            registration_method="mutual_information",
        )

        acquisition_tool.acquire_image(
            y_center=184, x_center=184, size_y=128, size_x=128
        )
        acquisition_tool.acquire_image(
            y_center=164, x_center=164, size_y=128, size_x=128
        )

        offset = self.register_previous_current(acquisition_tool, registration_tool)

        if self.debug:
            print("Offset (mutual information): ", offset)
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(acquisition_tool.image_k)
            axs[0].set_title(f"Image {acquisition_tool.counter_acquire_image}")
            axs[1].imshow(acquisition_tool.image_km1)
            axs[1].set_title(f"Image {acquisition_tool.counter_acquire_image - 1}")
            plt.show()

        assert np.allclose(offset, [-20, -20], atol=1.0)
        return

    def test_image_registration_ncc(self):
        np.random.seed(123)

        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )

        acquisition_tool = SimulatedAcquireImage(whole_image, return_message=False)
        registration_tool = ImageRegistration(
            registration_method="ncc",
            registration_algorithm_kwargs={"max_shift": 25},
        )

        acquisition_tool.acquire_image(
            y_center=184, x_center=184, size_y=128, size_x=128
        )
        acquisition_tool.acquire_image(
            y_center=164, x_center=164, size_y=128, size_x=128
        )

        previous_info = acquisition_tool.get_previous_image_info()
        current_info = acquisition_tool.get_current_image_info()
        offset = registration_tool.register_images(
            image_t=acquisition_tool.image_k,
            image_r=acquisition_tool.image_km1,
            psize_t=current_info["psize"],
            psize_r=previous_info["psize"],
            registration_algorithm_kwargs={"max_shift": 25},
        )

        assert np.allclose(offset, [-20, -20])
        return

    def test_image_registration_from_buffer_arrays(self):
        np.random.seed(123)

        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )

        acquisition_tool = SimulatedAcquireImage(whole_image, return_message=True)
        registration_tool = ImageRegistration(registration_method="phase_correlation")

        acquisition_results = []
        acquisition_results.append(acquisition_tool.acquire_image(
            y_center=184, x_center=184, size_y=128, size_x=128
        ))
        acquisition_results.append(acquisition_tool.acquire_image(
            y_center=164, x_center=164, size_y=128, size_x=128
        ))

        try:
            offset = self.register_previous_current(acquisition_tool, registration_tool)

            assert np.allclose(offset, [-20, -20])
        finally:
            for acquisition_result in acquisition_results:
                for path in (
                    acquisition_result["img_path"],
                    acquisition_result["raw_data_path"],
                ):
                    if os.path.exists(path):
                        os.remove(path)
        return

    def test_image_registration_from_dumped_array_paths(self):
        np.random.seed(123)

        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )

        acquisition_tool = SimulatedAcquireImage(whole_image, return_message=True)
        registration_tool = ImageRegistration(registration_method="phase_correlation")

        acquisition_results = []
        acquisition_results.append(acquisition_tool.acquire_image(
            y_center=184, x_center=184, size_y=128, size_x=128
        ))
        acquisition_results.append(acquisition_tool.acquire_image(
            y_center=164, x_center=164, size_y=128, size_x=128
        ))
        current_path = None
        previous_path = None

        try:
            current_info = acquisition_tool.get_current_image_info()
            previous_info = acquisition_tool.get_previous_image_info()
            current_path = acquisition_tool.dump_array("image_k")["array_path"]
            previous_path = acquisition_tool.dump_array("image_km1")["array_path"]
            offset = registration_tool.get_offset_from_paths(
                current_image_path=current_path,
                reference_image_path=previous_path,
                current_pixel_size=current_info["psize"],
                reference_pixel_size=previous_info["psize"],
            )

            assert np.allclose(offset, [-20, -20])
        finally:
            for array_path in (current_path, previous_path):
                if array_path is not None and os.path.exists(array_path):
                    os.remove(array_path)
            for acquisition_result in acquisition_results:
                for path in (
                    acquisition_result["img_path"],
                    acquisition_result["raw_data_path"],
                ):
                    if os.path.exists(path):
                        os.remove(path)
        return
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestImageRegistrationTool()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_image_registration()
    tester.test_image_registration_diff_size()
    tester.test_image_registration_mutual_information()
