from typing import Optional, Literal
from textwrap import dedent
import logging

from eaa.tools.imaging.acquisition import AcquireImage
from eaa.tools.imaging.param_tuning import SetParameters
from eaa.task_managers.imaging.base import ImagingBaseTaskManager
from eaa.tools.base import ToolReturnType
from eaa.agents.base import print_message

logger = logging.getLogger(__name__)


class ParameterTuningTaskManager(ImagingBaseTaskManager):
    
    def __init__(
        self, 
        model_name: str = "gpt-4o", 
        model_base_url: str = None, 
        access_token: str = None,
        other_llm_config: Optional[dict] = None,
        api_type: Literal["openai", "asksage"] = "openai",
        param_setting_tool: SetParameters = None,
        acquisition_tool: AcquireImage = None,
        initial_parameters: dict[str, float] = None,
        parameter_ranges: list[tuple[float, ...], tuple[float, ...]] = None,
        message_db_path: Optional[str] = None,
        build: bool = True,
        *args, **kwargs
    ) -> None:
        """An agent that searches for the best setup parameters
        for an imaging system.

        Parameters
        ----------
        model_name : str, optional
            The name of the model to use.
        model_base_url : str, optional
            The base URL of the model. This is only needed for
            self-hosted models.
        access_token : str
            The access token or API key for the model.
        other_llm_config : Optional[dict]
            Other configuration for the model, not including the model name,
            base URL, and access token. This information is only needed when
            using an endpoint that requires them (such as AskSage). Keys in 
            this dictionary can include:
            - `cacert_path`: The path to the CA certificate file (*.pem).
            - `email`: The email of the user.
            - `user_base_url`: The user base URL of the endpoint (used by AskSage).
            - `server_base_url`: The server base URL for the endpoint (used by AskSage).
              When `api_type` is "asksage", this will be used as the model base URL and
              `model_base_url` will be ignored.
        api_type : Literal["openai", "asksage"]
            The type of the API format. This is determined by the API used by
            the inference endpoint. Use "openai" whenever the endpoint offers
            OpenAI-compatible API. For AskSage endpoints, use "asksage".
        param_setting_tool : SetParameters
            The tool to use to set the parameters.
        acquisition_tool : SimulatedAcquireImage, optional
            The tool to use to acquire images. This tool will 
            not be called by AI; it is executed automatically 
            following each parameter adjustment.
        initial_parameters : dict[str, float], optional
            The initial parameters given as a dictionary of 
            parameter names and values.
        parameter_ranges : list[tuple[float, ...], tuple[float, ...]]
            The ranges of the parameters. It should be given as a list of
            2 tuples, where the first tuple gives the lower bounds and the
            second tuple gives the upper bounds. The order of the parameters
            should match the order of the initial parameters.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path. This is essential
            if you want to use the WebUI, which polls the database
            for new messages.
        """
        if "tools" in kwargs.keys():
            raise ValueError(
                "`tools` should not be provided to `ParameterTuningTaskManager`. Instead, "
                "provide the `param_setting_tool` and `acquisition_tool`."
            )
            
        self.param_setting_tool = param_setting_tool
        self.acquisition_tool = acquisition_tool
        self.initial_parameters = initial_parameters
        self.parameter_names = list(initial_parameters.keys())
        self.parameter_ranges = parameter_ranges
        
        super().__init__(
            model_name=model_name, 
            model_base_url=model_base_url,
            access_token=access_token,
            other_llm_config=other_llm_config,
            api_type=api_type,
            tools=[param_setting_tool],
            message_db_path=message_db_path,
            build=build,
            *args, **kwargs
        )
        
    def set_initial_parameters(self, initial_params: dict[str, float]):
        self.initial_parameters = initial_params
        
    def prerun_check(self, *args, **kwargs) -> bool:
        if self.initial_parameters is None:
            raise ValueError("initial_parameters must be provided.")
        return super().prerun_check(*args, **kwargs)
        
    def run(
        self,
        acquisition_tool_kwargs: dict = {},
        n_past_images_to_keep: int = 3,
        max_iters: int = 10, 
        initial_prompt: Optional[str] = None,
        additional_prompt: Optional[str] = None,
    ) -> None:
        """Run the parameter tuning task.
        
        Parameters
        ----------
        acquisition_tool_kwargs : dict
            The arguments for the acquisition tool. These arguments will be
            used to acquire images for evaluation.
        n_past_images_to_keep : int, optional
            The number of most recent images to keep in the context. Having past
            images in the context allows to agent to "remember" images it
            has seen before; however, it also increases the context size
            and inference cost.
        max_iters : int, optional
            The maximum number of iterations to run.
        initial_prompt : str, optional
            If provided, this prompt will override the default initial prompt.
        additional_prompt : str, optional
            If provided, this prompt will be added to the initial prompt (either
            the default one or the one provided by `initial_prompt`).
        """
        self.prerun_check()
        
        initial_parameter_values = list(self.initial_parameters.values())
        self.param_setting_tool.set_parameters(initial_parameter_values)
        last_img_path = self.acquisition_tool.acquire_image(**acquisition_tool_kwargs)
        
        bounds_str = ""
        for i, param in enumerate(self.parameter_names):
            bounds_str += f"{param}: {self.parameter_ranges[0][i]} to {self.parameter_ranges[1][i]}\n"
        
        if initial_prompt is None:
            initial_prompt = dedent(
                f"""\
                You are tuning the parameters of a microscope to attain the best
                image sharpness. The parameters are {list(self.parameter_names)},
                and their current values are {initial_parameter_values}. An image acquired
                with the current parameters is shown below. 
                
                <img {last_img_path}>
                
                Here are the tunable ranges of the parameters:
                {bounds_str}
                
                You can change the parameters using your parameter setting tool. 
                An image acquired with the new parameters will be given to you
                after each parameter change. Here are some detailed instructions:
                
                - Tune parameters one by one. Start with the first parameter, tweak it
                to attain the sharpest possible image, then move on to the next parameter.
                Do not change more than one parameter at a time.
                - The sharpness of the image is convex with regards to the parameters. There
                is only one optimal point; assume there is no local maximum. As such, if
                you find the image comes more blurry when changing a parameter in a direction,
                you should consider changing it the other way; if you find the image comes
                sharper when changing a parameter in a direction, you are on the right track.
                - For each parameter, first get a coarse estimate of the optimal value, then
                fine-tune it. To get a coarse estimate, look for a peak in the sharpness. In
                other words, find a parameter value that gives a sharper image than the value
                immediately before and after it. For example, if the image becomes sharper when
                you increase the parameter from 4 to 5, but becomes blurrier when you increase
                it from 5 to 6, then the optimal value is around 5.
                - Choose the step size for changing parameters wisely. For each parameter, start
                with a large step size, and decrease it as you get closer to the optimal point.
                - Only call the parameter setting tool one at a time. Do not call it multiple times
                in one response.
                
                When you finish or when you need human input, add "TERMINATE" to your response.\
                """
            )
        if additional_prompt is not None:
            initial_prompt += "\nAdditional instructions:\n" + additional_prompt
        
        round = 0
        response, outgoing = self.agent.receive(
            initial_prompt,
            context=self.context,
            image_path=last_img_path,
            return_outgoing_message=True
        )
        self.update_message_history(outgoing, update_context=True, update_full_history=True)
        self.update_message_history(response, update_context=True, update_full_history=True)
        while round < max_iters:
            if response["content"] is not None and "TERMINATE" in response["content"]:
                message = self.get_user_input(
                    "Termination condition triggered. What to do next? Type \"exit\" to exit. "
                )
                if message.lower() == "exit":
                    return
                else:
                    response, outgoing = self.agent.receive(
                        message,
                        context=self.context,
                        image_path=None,
                        return_outgoing_message=True
                    )
                    self.update_message_history(outgoing, update_context=True, update_full_history=True)
                    self.update_message_history(response, update_context=True, update_full_history=True)
                    continue
            
            tool_responses, tool_response_types = self.agent.handle_tool_call(response, return_tool_return_types=True)
            if len(tool_responses) == 1:
                tool_response = tool_responses[0]
                tool_response_type = tool_response_types[0]
                # Just save the tool response, but don't send yet. We will send it
                # together with the image later.
                print_message(tool_response)
                self.update_message_history(tool_response, update_context=True, update_full_history=True)
                
                if tool_response_type == ToolReturnType.EXCEPTION:
                    response, outgoing = self.agent.receive(
                        "The tool returned an exception. Please fix the exception and try again.",
                        image_path=None,
                        context=self.context,
                        return_outgoing_message=True
                    )
                else:
                    # Acquire an image with the new parameters.
                    last_img_path = self.acquisition_tool.acquire_image(**acquisition_tool_kwargs)
                    response, outgoing = self.agent.receive(
                        "An image acquired with the new parameters is shown below.",
                        image_path=last_img_path,
                        context=self.context,
                        return_outgoing_message=True
                    )
                self.purge_context_images(keep_fist_n=1, keep_last_n=n_past_images_to_keep - 1)
                self.update_message_history(outgoing, update_context=True, update_full_history=True)
                self.update_message_history(response, update_context=True, update_full_history=True)
            elif len(tool_responses) > 1:
                response, outgoing = self.agent.receive(
                    "There are more than one tool calls in your response. "
                    "Make sure you only make one call at a time. Please redo "
                    "your tool calls.",
                    image_path=None,
                    context=self.context,
                    return_outgoing_message=True
                )
                self.update_message_history(outgoing, update_context=True, update_full_history=True)
                self.update_message_history(response, update_context=True, update_full_history=True)
            else:
                response, outgoing = self.agent.receive(
                    "There is no tool call in the response. Make sure you call the tool correctly.",
                    image_path=None,
                    context=self.context,
                    return_outgoing_message=True
                )
                self.update_message_history(outgoing, update_context=True, update_full_history=True)
                self.update_message_history(response, update_context=True, update_full_history=True)
            round += 1
