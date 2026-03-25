# API Reference

## `ScanningMicroscopeFocusingTaskManager.__init__`
```python
def __init__(
    self,
    llm_config: LLMConfig = None,
    memory_config: Optional[MemoryManagerConfig] = None,
    param_setting_tool: SetParameters = None,
    acquisition_tool: AcquireImage = None,
    image_registration_tool: Optional[ImageRegistration] = None,
    additional_tools: list[BaseTool] = (),
    initial_parameters: dict[str, float] = None,
    parameter_ranges: list[tuple[float, ...], tuple[float, ...]] = None,
    use_feature_tracking_subtask: bool = False,
    feature_tracking_kwargs: Optional[dict] = None,
    session_db_path: Optional[str] = "messages.db",
    build: bool = True,
    *args,
    **kwargs,
) -> None:
```

Initializes a focusing task manager that alternates line scans, parameter updates, and imaging.

- `llm_config`: LLM configuration to use.
- `memory_config`: Optional memory configuration forwarded to the agent.
- `param_setting_tool`: Tool to set optics parameters.
- `acquisition_tool`: Tool for 2D image acquisition and line scans.
- `image_registration_tool`: Optional registration tool for drift handling.
- `additional_tools`: Extra tools available to the agent.
- `initial_parameters`: Initial parameter values.
- `parameter_ranges`: Parameter bounds for tuning.
- `use_feature_tracking_subtask`: Whether to enable feature tracking when drift occurs.
- `feature_tracking_kwargs`: Keyword args for feature tracking (required if enabled).
- `session_db_path`: Optional SQLite path for storing chat history.
- `build`: Whether to build internal state immediately.

## `ScanningMicroscopeFocusingTaskManager.run`
```python
def run(
    self,
    reference_image_path: Optional[str] = None,
    reference_feature_description: Optional[str] = None,
    suggested_2d_scan_kwargs: dict = None,
    suggested_parameter_step_size: Optional[float] = None,
    line_scan_step_size: float = None,
    use_registration_in_workflow: bool = True,
    add_reference_image_to_images_acquired: bool = False,
    initial_prompt: Optional[str] = None,
    max_iters: int = 20,
    n_last_images_to_keep_in_context: Optional[int] = None,
    additional_prompt: Optional[str] = None,
    *args,
    **kwargs,
):
```

Runs the LLM-guided focusing workflow and iteratively improves focus based on line scans.

- `reference_image_path`: Path to a reference image with the target feature.
- `reference_feature_description`: Text description of the focus feature (ignored if reference image is provided).
- `suggested_2d_scan_kwargs`: Suggested kwargs for 2D image acquisition.
- `suggested_parameter_step_size`: Suggested step size for parameter adjustments.
- `line_scan_step_size`: Step size for line scans.
- `use_registration_in_workflow`: Whether to register consecutive images.
- `add_reference_image_to_images_acquired`: Stitch reference image into acquisitions.
- `initial_prompt`: Override default prompt.
- `max_iters`: Maximum number of iterations.
- `n_last_images_to_keep_in_context`: Number of recent images to keep in context.
- `additional_prompt`: Extra instructions appended to the prompt.
