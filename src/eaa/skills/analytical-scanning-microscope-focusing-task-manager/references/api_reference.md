# API Reference

## `AnalyticalScanningMicroscopeFocusingTaskManager.__init__`
```python
def __init__(
    self,
    llm_config: LLMConfig = None,
    memory_config: Optional[MemoryManagerConfig] = None,
    param_setting_tool: SetParameters = None,
    acquisition_tool: AcquireImage = None,
    initial_parameters: dict[str, float] = None,
    parameter_ranges: list[tuple[float, ...], tuple[float, ...]] = None,
    session_db_path: Optional[str] = "messages.db",
    build: bool = True,
    line_scan_tool_x_coordinate_args: Tuple[str, ...] = ("x_center",),
    line_scan_tool_y_coordinate_args: Tuple[str, ...] = ("y_center",),
    image_acquisition_tool_x_coordinate_args: Tuple[str, ...] = ("x_center",),
    image_acquisition_tool_y_coordinate_args: Tuple[str, ...] = ("y_center",),
    *args,
    **kwargs,
) -> None:
```

Builds an analytical focusing manager that uses registration and Bayesian optimization.

- `llm_config`: LLM configuration to use.
- `memory_config`: Optional memory configuration forwarded to the agent.
- `param_setting_tool`: Tool to set optics parameters.
- `acquisition_tool`: Tool for 2D image acquisition and line scans.
- `initial_parameters`: Initial parameter values.
- `parameter_ranges`: Parameter bounds for tuning.
- `session_db_path`: Optional SQLite path for storing chat history.
- `build`: Whether to build internal state immediately.
- `line_scan_tool_x_coordinate_args`: Line scan args for x-coordinate updates.
- `line_scan_tool_y_coordinate_args`: Line scan args for y-coordinate updates.
- `image_acquisition_tool_x_coordinate_args`: Image acquisition args for x-coordinate updates.
- `image_acquisition_tool_y_coordinate_args`: Image acquisition args for y-coordinate updates.

## `AnalyticalScanningMicroscopeFocusingTaskManager.run`
```python
def run(
    self,
    initial_2d_scan_kwargs: dict = None,
    initial_line_scan_kwargs: dict = None,
    n_initial_points: int = 5,
    initial_sampling_window_size: Optional[Tuple[float, ...]] = None,
    n_max_bo_iterations: int = 99,
    parameter_change_step_limit: Optional[float | Tuple[float, ...]] = None,
    termination_behavior: Literal["ask", "return"] = "ask",
    *args,
    **kwargs,
):
```

Runs the analytical focusing workflow with initial measurements and Bayesian optimization.

- `initial_2d_scan_kwargs`: Keyword args for the initial 2D scan.
- `initial_line_scan_kwargs`: Keyword args for the initial line scan.
- `n_initial_points`: Number of initial measurements to seed BO.
- `initial_sampling_window_size`: Window size for initial sampling.
- `n_max_bo_iterations`: Maximum number of BO iterations.
- `parameter_change_step_limit`: Limit on parameter change per step.
- `termination_behavior`: Whether to ask or return when max iterations reached.
