---
name: bayesian-optimization-task-manager
description: Bayesian optimization loop that evaluates an objective and updates a BO tool.
---

# Bayesian Optimization Task Manager

## Overview
`BayesianOptimizationTaskManager` runs a Bayesian optimization loop using a provided
`BayesianOptimizationTool` and objective function.

## Task Manager Interface
Pass a BO tool and objective function, then call `run()` with a chosen number of iterations.

See `references/api_reference.md` for full signatures.

## Example
```python
import torch

from eaa.task_manager.tuning.bo import BayesianOptimizationTaskManager
from eaa.tool.bo import BayesianOptimizationTool

bo_tool = BayesianOptimizationTool(bounds=([0.0, 0.0], [1.0, 1.0]))

def objective_function(x: torch.Tensor) -> torch.Tensor:
    return (1.0 - (x - 0.5).pow(2).sum(dim=1, keepdim=True))

task_manager = BayesianOptimizationTaskManager(
    bayesian_optimization_tool=bo_tool,
    objective_function=objective_function,
    n_initial_points=5,
    message_db_path="messages.db",
)

task_manager.run(n_iterations=10)
```
