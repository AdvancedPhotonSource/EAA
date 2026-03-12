# API Reference

## `AnalyticalFeatureTrackingTaskManager.__init__`
```python
def __init__(
    self,
    llm_config: LLMConfig = None,
    memory_config: Optional[MemoryManagerConfig] = None,
    image_acquisition_tool: AcquireImage = None,
    session_db_path: Optional[str] = "messages.db",
    build: bool = True,
    image_acquisition_tool_x_coordinate_args: Tuple[str, ...] = ("x_center",),
    image_acquisition_tool_y_coordinate_args: Tuple[str, ...] = ("y_center",),
    *args,
    **kwargs,
) -> None:
```

Builds an analytical feature tracking manager that searches in a spiral pattern.

- `llm_config`: Configuration for the LLM (required for feature presence detection).
- `memory_config`: Optional memory configuration forwarded to the agent.
- `image_acquisition_tool`: Acquisition tool used to collect images (required).
- `session_db_path`: Optional SQLite path for storing chat history.
- `build`: Whether to build internal state immediately.
- `image_acquisition_tool_x_coordinate_args`: Argument names used for x-coordinate updates.
- `image_acquisition_tool_y_coordinate_args`: Argument names used for y-coordinate updates.

## `AnalyticalFeatureTrackingTaskManager.run`
```python
def run(
    self,
    current_acquisition_kwargs: dict,
    reference_image: np.ndarray,
    step_size: Tuple[float, float],
    reference_image_pixel_size: float = 1.0,
    n_max_rounds: int = 20,
) -> np.ndarray:
```

Runs a spiral search for the feature and returns the (y, x) offset needed to align the FOV.

- `current_acquisition_kwargs`: Current acquisition tool kwargs.
- `reference_image`: 2D numpy array of the reference feature image.
- `step_size`: Spiral step size (y, x).
- `reference_image_pixel_size`: Pixel size of the reference image.
- `n_max_rounds`: Maximum search rounds.
