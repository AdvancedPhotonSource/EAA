import os
import argparse

import tifffile
import numpy as np

from eaa_core.tool.base import BaseTool
from eaa_imaging.tool.imaging.acquisition import SimulatedAcquireImage

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

    def test_simulated_image_message_returns_json_payload(self):
        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )

        tool = SimulatedAcquireImage(whole_image, return_message=True)
        result = tool.acquire_image(y_center=228, x_center=228, size_y=64, size_x=64)

        assert isinstance(result, dict)
        assert isinstance(result.get("img_path"), str)
        assert "array_path" not in result
        assert result["psize"] == 1

    def test_simulated_image_buffers_and_payloads_are_available(self):
        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )

        tool = SimulatedAcquireImage(whole_image, return_message=True)
        tool.acquire_image(y_center=228, x_center=228, size_y=64, size_x=64)
        first_image = tool.image_k
        tool.acquire_image(y_center=230, x_center=230, size_y=64, size_x=64)
        second_image = tool.image_k
        tool.acquire_image(y_center=232, x_center=232, size_y=64, size_x=64)
        third_image = tool.image_k
        tool.acquire_image(y_center=234, x_center=234, size_y=64, size_x=64)
        current_info = tool.get_current_image_info()
        previous_info = tool.get_previous_image_info()
        initial_info = tool.get_initial_image_info()
        payload = tool.get_attribute_payload("image_km1")

        assert np.array_equal(tool.image_0, first_image)
        assert not np.array_equal(tool.image_km1, second_image)
        assert np.array_equal(tool.image_km1, third_image)
        assert current_info["shape"] == [64, 64]
        assert previous_info["shape"] == [64, 64]
        assert initial_info["shape"] == [64, 64]
        assert "array_path" not in current_info
        assert np.array_equal(BaseTool.decode_array_payload(payload), third_image)

    def test_dump_array_saves_requested_buffer(self):
        whole_image = tifffile.imread(
            os.path.join(
                self.get_ci_input_data_dir(),
                'simulated_images',
                'cameraman.tiff'
            )
        )

        tool = SimulatedAcquireImage(whole_image, return_message=True)
        tool.acquire_image(y_center=228, x_center=228, size_y=64, size_x=64)
        tool.acquire_image(y_center=230, x_center=230, size_y=64, size_x=64)

        result = tool.dump_array("image_km1")

        try:
            assert isinstance(result["array_path"], str)
            assert os.path.exists(result["array_path"])
            assert np.array_equal(
                np.load(result["array_path"], allow_pickle=False),
                tool.image_km1,
            )
        finally:
            array_path = result.get("array_path")
            if isinstance(array_path, str):
                os.remove(array_path)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    tester = TestSimulatedImageAcquisition()
    tester.setup_method(name="", generate_data=False, generate_gold=args.generate_gold, debug=True)
    tester.test_simulated_image_acquisition()
    tester.test_advanced_simulated_image_acquisition()
    tester.test_simulated_line_scan()
        
