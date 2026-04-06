---
name: scanning-microscope-focusing-task-manager
description: This document describes how to launch an LLM-guided focusing task manager that alternates line scans, parameter updates, and imaging.
---

# Scanning Microscope Focusing Task Manager

## Overview
`ScanningMicroscopeFocusingTaskManager` orchestrates a focusing workflow by alternating line scans,
parameter adjustments, and image acquisition to minimize feature width and improve sharpness.

## Task Manager Interface
Provide parameter-setting and acquisition tools, parameter bounds, and optional feature tracking settings.
Call `run()` to execute the LLM-guided focusing loop.

See `references/api_reference.md` for full signatures.

## Example
```python
import numpy as np

from eaa_core.api.llm_config import OpenAIConfig
from eaa_imaging.task_manager.tuning.focusing import ScanningMicroscopeFocusingTaskManager
from eaa_imaging.tool.imaging.acquisition import SimulatedAcquireImage
from eaa_imaging.tool.imaging.param_tuning import SimulatedSetParameters

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

task_manager = ScanningMicroscopeFocusingTaskManager(
    llm_config=OpenAIConfig(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    param_setting_tool=parameter_tool,
    acquisition_tool=acquisition_tool,
    initial_parameters={"focus": 0.0, "astigmatism": 0.0},
    parameter_ranges=parameter_ranges,
    session_db_path="messages.db",
)

task_manager.run(
    reference_feature_description="bright diagonal line",
    suggested_2d_scan_kwargs={"loc_x": 0.0, "loc_y": 0.0, "size_x": 64, "size_y": 64},
    line_scan_step_size=1.0,
    max_iters=5,
)
```
