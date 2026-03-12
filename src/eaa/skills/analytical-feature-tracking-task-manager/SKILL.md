---
name: analytical-feature-tracking-task-manager
description: Runs a spiral search to align a reference feature analytically using acquisition and registration.
---

# Analytical Feature Tracking Task Manager

## Overview
`AnalyticalFeatureTrackingTaskManager` performs a spiral search without an LLM-driven loop to align a
reference feature. It evaluates offsets and returns the displacement needed to re-center the feature.

## Task Manager Interface
Call `run()` with acquisition kwargs, a reference image array, and a step size to obtain the offset.

See `references/api_reference.md` for full signatures.

## Example
```python
import numpy as np

from eaa.api.llm_config import OpenAIConfig
from eaa.task_manager.imaging.analytical_feature_tracking import (
    AnalyticalFeatureTrackingTaskManager,
)
from eaa.tool.imaging.acquisition import SimulatedAcquireImage

whole_image = np.zeros((256, 256))
acquisition_tool = SimulatedAcquireImage(whole_image=whole_image)

task_manager = AnalyticalFeatureTrackingTaskManager(
    llm_config=OpenAIConfig(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    image_acquisition_tool=acquisition_tool,
    image_acquisition_tool_x_coordinate_args=("loc_x",),
    image_acquisition_tool_y_coordinate_args=("loc_y",),
    session_db_path="messages.db",
)

offset = task_manager.run(
    current_acquisition_kwargs={"loc_x": 0.0, "loc_y": 0.0, "size_x": 64, "size_y": 64},
    reference_image=np.zeros((64, 64)),
    step_size=(2.0, 2.0),
)
print(offset)
```
