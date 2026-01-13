# API Reference

## `MicroscopyOpticsTuningBOTaskManager.__init__`
```python
def __init__(
    self,
    llm_config: LLMConfig = None,
    memory_config: Optional[MemoryManagerConfig] = None,
    image_acquisition_tool: BaseTool = None,
    parameter_setting_tool: BaseTool = None,
    bayesian_optimization_tool: BayesianOptimizationTool = None,
    additional_tools: list[BaseTool] = (),
    initial_points: Optional[torch.Tensor] = None,
    n_initial_points: int = 20,
    image_acquisition_kwargs: dict = {},
    feature_tracking_kwargs: dict = {},
    message_db_path: Optional[str] = "messages.db",
    *args,
    **kwargs,
):
```

Creates a task manager that combines Bayesian optimization with feature tracking for optics tuning.

- `llm_config`: Optional LLM configuration.
- `memory_config`: Optional memory configuration forwarded to the underlying agents.
- `image_acquisition_tool`: Tool for image acquisition (required).
- `parameter_setting_tool`: Tool for setting parameters (required).
- `bayesian_optimization_tool`: BO tool for proposing parameters (required).
- `additional_tools`: Extra tools available to the agent.
- `initial_points`: Optional tensor of initial BO points.
- `n_initial_points`: Number of random initial points if none are supplied.
- `image_acquisition_kwargs`: Arguments for acquiring images during evaluation.
- `feature_tracking_kwargs`: Args for the feature tracking subtask.
- `message_db_path`: Optional SQLite path for storing chat history.

## `MicroscopyOpticsTuningBOTaskManager.run`
```python
def run(
    self,
    n_iterations: int = 50,
    *args,
    **kwargs,
) -> None:
```

Runs the Bayesian optimization loop, invoking feature tracking and image acquisition per iteration.

- `n_iterations`: Number of optimization iterations to execute.
