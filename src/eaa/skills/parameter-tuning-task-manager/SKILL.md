---
name: parameter-tuning-task-manager
description: Iterative parameter tuning workflow that adjusts optics and evaluates images.
---

# Parameter Tuning Task Manager

## Overview
`ParameterTuningTaskManager` iteratively adjusts optics parameters and evaluates the resulting
images, keeping recent images in context to guide the tuning strategy.

## Task Manager Interface
Provide parameter-setting and acquisition tools, initial parameters, and valid ranges.
Run `run()` with acquisition kwargs to start the tuning loop.

See `references/api_reference.md` for full signatures.

## Example
```python
import numpy as np

from eaa.api.llm_config import OpenAIConfig
from eaa.task_manager.tuning.focusing import ParameterTuningTaskManager
from eaa.tool.imaging.acquisition import SimulatedAcquireImage
from eaa.tool.imaging.param_tuning import SimulatedSetParameters

whole_image = np.zeros((256, 256))
acquisition_tool = SimulatedAcquireImage(whole_image=whole_image)
parameter_names = ["focus", "astigmatism"]
parameter_ranges = [(0.0, 0.0), (1.0, 1.0)]
parameter_tool = SimulatedSetParameters(
    acquisition_tool=acquisition_tool,
    parameter_names=parameter_names,
    parameter_ranges=parameter_ranges,
    true_parameters=[0.5, 0.5],
)

task_manager = ParameterTuningTaskManager(
    llm_config=OpenAIConfig(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    param_setting_tool=parameter_tool,
    acquisition_tool=acquisition_tool,
    initial_parameters={"focus": 0.0, "astigmatism": 0.0},
    parameter_ranges=parameter_ranges,
    message_db_path="messages.db",
)

task_manager.run(
    acquisition_tool_kwargs={"loc_x": 0.0, "loc_y": 0.0, "size_x": 64, "size_y": 64},
    max_iters=5,
)
```
