---
name: analytical-scanning-microscope-focusing-task-manager
description: Analytical focusing workflow that mixes registration, feature tracking, and Bayesian optimization.
---

# Analytical Scanning Microscope Focusing Task Manager

## Overview
`AnalyticalScanningMicroscopeFocusingTaskManager` focuses a scanning microscope using analytical logic.
It collects initial measurements, performs registration or feature tracking, and then uses Bayesian
optimization to refine parameters.

## Task Manager Interface
Provide parameter-setting and acquisition tools, parameter ranges, and initial scan kwargs.
Call `run()` to execute the analytical focusing loop.

See `references/api_reference.md` for full signatures.

## Example
```python
import numpy as np

from eaa.api.llm_config import OpenAIConfig
from eaa.task_manager.tuning.analytical_focusing import (
    AnalyticalScanningMicroscopeFocusingTaskManager,
)
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

task_manager = AnalyticalScanningMicroscopeFocusingTaskManager(
    llm_config=OpenAIConfig(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    param_setting_tool=parameter_tool,
    acquisition_tool=acquisition_tool,
    initial_parameters={"focus": 0.0, "astigmatism": 0.0},
    parameter_ranges=parameter_ranges,
    message_db_path="messages.db",
)

task_manager.run(
    initial_2d_scan_kwargs={"loc_x": 0.0, "loc_y": 0.0, "size_x": 64, "size_y": 64},
    initial_line_scan_kwargs={"start_x": 0.0, "start_y": 0.0, "end_x": 64.0, "end_y": 0.0, "step": 1.0},
    n_initial_points=3,
    initial_sampling_window_size=(0.2, 0.2),
    n_max_bo_iterations=5,
)
```
