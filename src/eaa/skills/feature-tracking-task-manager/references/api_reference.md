# API Reference

## `ROISearchTaskManager.__init__`
```python
def __init__(
    self,
    llm_config: LLMConfig = None,
    memory_config: Optional[MemoryManagerConfig] = None,
    image_acquisition_tool: AcquireImage = None,
    image_registration_tool: ImageRegistration = None,
    additional_tools: list[BaseTool] = (),
    session_db_path: Optional[str] = None,
    build: bool = True,
    *args,
    **kwargs,
) -> None:
```

Creates an ROI-search task manager that acquires images while searching for a target feature.

- `llm_config`: Configuration for the LLM.
- `memory_config`: Optional memory configuration forwarded to the task manager.
- `image_acquisition_tool`: Acquisition tool used to collect images.
- `image_registration_tool`: Optional registration tool available during the workflow.
- `additional_tools`: Extra tools available to the task manager.
- `session_db_path`: Optional SQLite path for storing chat history.
- `build`: Whether to build internal state immediately.

## `ROISearchTaskManager.run`
```python
def run(
    self,
    feature_description: str = None,
    y_range: tuple[float, float] = None,
    x_range: tuple[float, float] = None,
    fov_size: tuple[float, float] = None,
    step_size: tuple[float, float] = None,
    max_rounds: int = 99,
    n_first_images_to_keep_in_context: Optional[int] = None,
    n_last_images_to_keep_in_context: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    additional_prompt: Optional[str] = None,
    *args,
    **kwargs,
) -> None:
```

Runs an LLM-guided search for the best field of view containing a described feature.

- `feature_description`: Text or image-tag description of the feature to locate.
- `y_range`/`x_range`: Search bounds for the field of view.
- `fov_size`: Initial field-of-view size (height, width).
- `step_size`: Grid-search step sizes (y, x).
- `max_rounds`: Max tool-call rounds.
- `n_first_images_to_keep_in_context`: Keep first N images in context.
- `n_last_images_to_keep_in_context`: Keep last N images in context.
- `initial_prompt`: Override default prompt.
- `additional_prompt`: Additional instructions appended to the prompt.

## `FeatureTrackingTaskManager.__init__`
```python
def __init__(
    self,
    llm_config: LLMConfig = None,
    memory_config: Optional[MemoryManagerConfig] = None,
    image_acquisition_tool: AcquireImage = None,
    image_registration_tool: ImageRegistration = None,
    additional_tools: list[BaseTool] = (),
    session_db_path: Optional[str] = None,
    build: bool = True,
    *args,
    **kwargs,
) -> None:
```

Creates a feature-tracking task manager that acquires and registers images to re-center a target feature.

- `llm_config`: Configuration for the LLM.
- `memory_config`: Optional memory configuration forwarded to the task manager.
- `image_acquisition_tool`: Acquisition tool used to collect images.
- `image_registration_tool`: Registration tool for aligning images.
- `additional_tools`: Extra tools available to the task manager.
- `session_db_path`: Optional SQLite path for storing chat history.
- `build`: Whether to build internal state immediately.

## `FeatureTrackingTaskManager.run`
```python
def run(
    self,
    reference_image_path: Optional[str] = None,
    initial_position: Optional[tuple[float, float]] = None,
    initial_fov_size: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    x_range: Optional[tuple[float, float]] = None,
    add_reference_image_to_images_acquired: bool = False,
    max_rounds: int = 99,
    n_first_images_to_keep_in_context: Optional[int] = None,
    n_last_images_to_keep_in_context: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    additional_prompt: Optional[str] = None,
    termination_behavior: Literal["ask", "return"] = "ask",
    max_arounds_reached_behavior: Literal["return", "raise"] = "return",
) -> None:
```

Searches for a drifted feature using a reference image and returns it to the field of view.

- `reference_image_path`: Path to a reference image containing the feature.
- `initial_position`: Starting FOV position (y, x).
- `initial_fov_size`: Starting FOV size (height, width).
- `y_range`/`x_range`: Search bounds for the field of view.
- `add_reference_image_to_images_acquired`: Whether to stitch reference with acquisitions.
- `max_rounds`: Max tool-call rounds.
- `n_first_images_to_keep_in_context`: Keep first N images in context.
- `n_last_images_to_keep_in_context`: Keep last N images in context.
- `initial_prompt`: Override default prompt.
- `additional_prompt`: Additional instructions appended to the prompt.
- `termination_behavior`: Whether to ask or return at termination.
- `max_arounds_reached_behavior`: Whether to return or raise when max rounds are reached.
