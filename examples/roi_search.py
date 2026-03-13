from skimage import data

from eaa.task_manager.imaging.roi_search import ROISearchTaskManager
from eaa.tool.imaging.acquisition import SimulatedAcquireImage
from eaa.api.llm_config import OpenAIConfig


def main() -> None:
    """Run a minimal ROI-search example on the skimage cameraman image."""
    llm_config = OpenAIConfig(
        ...  # Replace with your LLMConfig instance.
    )

    whole_image = data.camera()
    acquisition_tool = SimulatedAcquireImage(
        whole_image=whole_image,
        add_axis_ticks=True,
        show_image_in_real_time=False,
    )

    task_manager = ROISearchTaskManager(
        llm_config=llm_config,
        image_acquisition_tool=acquisition_tool,
        use_webui=True,
    )
    task_manager.run(
        feature_description="the camera",
        y_range=(32.0, float(whole_image.shape[0])),
        x_range=(32.0, float(whole_image.shape[1])),
        fov_size=(64.0, 64.0),
        step_size=(64.0, 64.0),
    )


if __name__ == "__main__":
    main()
