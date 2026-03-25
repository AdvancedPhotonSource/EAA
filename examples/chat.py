from skimage import data

from eaa_core.task_manager.base import BaseTaskManager
from eaa_imaging.tool.imaging.acquisition import SimulatedAcquireImage
from eaa_core.api.llm_config import OpenAIConfig


def main() -> None:
    """Run a minimal chat example with a simulated image-acquisition tool."""
    llm_config = OpenAIConfig(
        ...  # Replace with your LLMConfig instance.
    )

    acquisition_tool = SimulatedAcquireImage(
        whole_image=data.camera(),
        add_axis_ticks=True,
        show_image_in_real_time=False,
    )

    task_manager = BaseTaskManager(
        llm_config=llm_config,
        tools=[acquisition_tool],
        use_webui=True,
    )
    task_manager.run_conversation()


if __name__ == "__main__":
    main()
