# API Reference

## `ParameterTuningTaskManager.__init__`
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
    *args,
    **kwargs,
) -> None:
```

Creates a parameter tuning task manager that updates optics parameters and evaluates images.

- `llm_config`: Configuration for the LLM.
- `memory_config`: Optional memory configuration forwarded to the agent.
- `param_setting_tool`: Tool to set optics parameters.
- `acquisition_tool`: Tool for image acquisition between parameter updates.
- `initial_parameters`: Initial parameter values.
- `parameter_ranges`: Parameter bounds for tuning.
- `session_db_path`: Optional SQLite path for storing chat history.
- `build`: Whether to build internal state immediately.

## `ParameterTuningTaskManager.run`
```python
def run(
    self,
    acquisition_tool_kwargs: dict = {},
    n_last_images_to_keep: int = 3,
    max_iters: int = 10,
    initial_prompt: Optional[str] = None,
    additional_prompt: Optional[str] = None,
) -> None:
```

Runs the parameter tuning workflow by iteratively adjusting parameters and reviewing images.

- `acquisition_tool_kwargs`: Keyword args passed to the acquisition tool.
- `n_last_images_to_keep`: Number of recent images to keep in context.
- `max_iters`: Maximum number of iterations.
- `initial_prompt`: Override default prompt.
- `additional_prompt`: Extra instructions appended to the prompt.
