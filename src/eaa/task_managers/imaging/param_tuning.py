from textwrap import dedent
import logging

from eaa.tools.imaging.acquisition import AcquireImage
from eaa.tools.imaging.param_tuning import TuneOpticsParameters
from eaa.task_managers.imaging.base import ImagingBaseTaskManager
import eaa.image_proc as ip

logger = logging.getLogger(__name__)


class ParameterTuningTaskManager(ImagingBaseTaskManager):
    
    def __init__(
        self,
        model_name: str = "gpt-4o", 
        model_base_url: str = None,
        param_setting_tool: TuneOpticsParameters = None,
        acquisition_tool: AcquireImage = None,
        initial_parameters: dict[str, float] = None,
        *args, **kwargs
    ) -> None:
        """An agent that searches for the best setup parameters
        for an imaging system.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use.
        model_base_url : str, optional
            The base URL of the model. This is only needed for self-hosted models.
        tools : list[BaseTool], optional
            A list of tools given to the agent.
        acquisition_tool : SimulatedAcquireImage, optional
            The tool to use to acquire images. This tool will not be called by AI; it is
            executed automatically following each parameter adjustment.
        initial_parameters : dict[str, float], optional
            The initial parameters given as a dictionary of parameter names and values.
        speaker_selection_method : Literal["round_robin", "random", "auto"], optional
            The method to select the next speaker in the group chat.
            - "round_robin": select the next speaker in a round-robin fashion.
            - "random": select the next speaker randomly.
            - "auto": let the LLM decide the next speaker. Some models might have issues
              with suggesting the speaker in the right format when used as the group chat
              manager. In that case, use "round_robin" or "random" instead.
        """
        if "tools" in kwargs.keys():
            raise ValueError(
                "`tools` should not be provided to `ParameterTuningTaskManager`. Instead, "
                "provide the `param_setting_tool` and `acquisition_tool`."
            )
        
        super().__init__(
            model_name=model_name, 
            model_base_url=model_base_url,
            tools=[param_setting_tool],
            *args, **kwargs
        )
        self.param_setting_tool = param_setting_tool
        self.acquisition_tool = acquisition_tool
        self.last_img = None
        self.image_path = None
        self.initial_parameters = initial_parameters
        
    def prerun_check(self, *args, **kwargs) -> bool:
        if self.initial_parameters is None:
            raise ValueError("initial_parameters must be provided.")
        if self.acquisition_tool.return_message:
            logger.warning(
                "acquisition_tool should return the image itself, not a path. "
                "`return_message` has been overridden to `False`."
            )
            self.acquisition_tool.return_message = False
        return super().prerun_check(*args, **kwargs)
        
    def run(
        self, 
        test_acquisition_location: tuple[float, float],
        test_acquisition_size: tuple[float, float],
        max_iters: int = 10, 
    ) -> None:
        """Run the parameter tuning task.
        
        Parameters
        ----------
        test_acquisition_location : tuple[float, float]
            The y, x coordinates of the top left corner for the image acquisition
            used to test the current parameter values.
        test_acquisition_size : tuple[float, float]
            The height, width of the image acquisition used to test the current
            parameter values.
        max_iters : int, optional
            The maximum number of iterations to run.
        """
        self.prerun_check()
        
        self.param_setting_tool(
            param1=self.initial_parameters["param1"],
            param2=self.initial_parameters["param2"],
            param3=self.initial_parameters["param3"],
        )
        self.last_img = self.acquisition_tool(
            loc_y=test_acquisition_location[0],
            loc_x=test_acquisition_location[1],
            size_y=test_acquisition_size[0],
            size_x=test_acquisition_size[1],
        )
        
        for i_iter in range(max_iters):
            if i_iter == 0:
                message = dedent(f"""\
                    You are given a tool that changes the optics parameters
                    of the imaging system. The current parameters are:
                    {self.initial_parameters}.
                    Now, use your tool to change the parameters. Change only
                    one parameter at a time. Use a step size of 1. You will be
                    given the image acquired with the new parameters after you
                    set them.
                    """
                )
            else:
                message = dedent(f"""\
                    In the attached image, the left hand side is the image acquired
                    with the previous parameters, and the right hand side is the image
                    acquired with the new parameters.
                    
                    Previous parameters: {self.param_setting_tool.parameter_history[-2]}
                    New parameters: {self.param_setting_tool.parameter_history[-1]}
                    Compare the new image with the previous one.
                    - If the new image is sharper, you are going in the right direction.
                      Keep changing the parameter in that direction.
                    - If the new image is blurrier, you are going in the wrong direction.
                      Revert the parameter to the previous value and try the opposite direction.
                    - If the new image is sharp (little blurriness at boundaries of features),
                      you have reached the optimal parameters. Report the parameters, and add
                      "TERMINATE" to the end of your response.
                    In the first 2 cases, use your tool to change the parameters to the new values.
                    """
                )
            
            # Ask the LLM to change the parameters.
            # Message is stored in memory only when the message does not contain an image.
            response = self.agent.receive(
                message=message,
                image_path=self.image_path,
                store_message=self.image_path is None,
                store_response=True
            )
            if "TERMINATE" in response["content"]:
                break
            
            tool_response = self.agent.handle_tool_call(response)
            
            if tool_response is not None:
                self.agent.receive(
                    tool_response, 
                    role="tool", 
                    request_response=False,
                    store_message=True, 
                )
                # Acquire the image with the new parameters.
                new_image = self.acquisition_tool(
                    loc_y=test_acquisition_location[0],
                    loc_x=test_acquisition_location[1],
                    size_y=test_acquisition_size[0],
                    size_x=test_acquisition_size[1],
                )
                
                stitched_image = ip.stitch_images([self.last_img, new_image])
                self.last_img = new_image
                self.image_path = self.acquisition_tool.save_image_to_temp_dir(
                    stitched_image, "stitched_image.png".format(), add_timestamp=True
                )
            
        logger.info(f"Final parameters: {self.param_setting_tool.parameter_history[-1]}")
