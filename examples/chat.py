import os

from skimage import data

from eaa_core.api.llm_config import OpenAIConfig
from eaa_core.task_manager.base import BaseTaskManager
from eaa_imaging.tool.imaging.acquisition import SimulatedAcquireImage


def main() -> None:
    """Run a minimal chat example with a simulated image-acquisition tool."""
    llm_config = OpenAIConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    acquisition_tool = SimulatedAcquireImage(
        whole_image=data.camera(),
        add_axis_ticks=True,
    )

    task_manager = BaseTaskManager(
        llm_config=llm_config,
        tools=[acquisition_tool],
        use_webui=False,
    )
    task_manager.run_conversation()


if __name__ == "__main__":
    main()
