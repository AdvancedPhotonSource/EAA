# API Reference

## `BayesianOptimizationTaskManager.__init__`
```python
def __init__(
    self,
    llm_config: LLMConfig = None,
    memory_config: Optional[MemoryManagerConfig] = None,
    bayesian_optimization_tool: BayesianOptimizationTool = None,
    additional_tools: list[BaseTool] = (),
    initial_points: Optional[torch.Tensor] = None,
    n_initial_points: int = 20,
    objective_function: Callable = None,
    message_db_path: Optional[str] = "messages.db",
    build: bool = True,
    *args,
    **kwargs,
) -> None:
```

Initializes a Bayesian optimization task manager with a BO tool and objective function.

- `llm_config`: Optional LLM configuration.
- `memory_config`: Optional memory configuration forwarded to the agent.
- `bayesian_optimization_tool`: BO tool used to suggest and update points (required).
- `additional_tools`: Extra tools available to the agent.
- `initial_points`: Optional tensor of initial points.
- `n_initial_points`: Number of random initial points if none are supplied.
- `objective_function`: Callable objective function to evaluate (required).
- `message_db_path`: Optional SQLite path for storing chat history.
- `build`: Whether to build internal state immediately.

## `BayesianOptimizationTaskManager.run`
```python
def run(
    self,
    n_iterations: int = 50,
    *args,
    **kwargs,
) -> None:
```

Runs Bayesian optimization iterations, continuing from the last state if previously run.

- `n_iterations`: Number of optimization iterations to execute.
