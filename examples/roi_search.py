import os

from skimage import data

from eaa_core.api.llm_config import OpenAIConfig
from eaa_imaging.task_manager.imaging.roi_search import ROISearchTaskManager
from eaa_imaging.tool.imaging.acquisition import SimulatedAcquireImage


def main() -> None:
    """Run a minimal ROI-search example on the skimage cameraman image."""
    llm_config = OpenAIConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    whole_image = data.camera()
    acquisition_tool = SimulatedAcquireImage(
        whole_image=whole_image,
        add_axis_ticks=True,
    )

    task_manager = ROISearchTaskManager(
        llm_config=llm_config,
        image_acquisition_tool=acquisition_tool,
        use_webui=False,
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
