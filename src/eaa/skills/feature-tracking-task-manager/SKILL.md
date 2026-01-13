---
name: feature-tracking-task-manager
description: Tracks or searches for sample features with image acquisition and registration tools.
---

# Feature Tracking Task Manager

## Overview
`FeatureTrackingTaskManager` guides an agent to locate or re-center a feature in a microscope field of view.
It uses acquisition and registration tools to move the FOV until the feature is centered.

## Task Manager Interface
This task manager provides `run_fov_search()` for initial discovery and `run_feature_tracking()` for
recovering drift based on a reference image.

See `references/api_reference.md` for full signatures.

## Example
```python
import numpy as np

from eaa.api.llm_config import OpenAIConfig
from eaa.task_manager.imaging.feature_tracking import FeatureTrackingTaskManager
from eaa.tool.imaging.acquisition import SimulatedAcquireImage
from eaa.tool.imaging.registration import ImageRegistration

whole_image = np.zeros((512, 512))
acquisition_tool = SimulatedAcquireImage(whole_image=whole_image)
registration_tool = ImageRegistration(image_acquisition_tool=acquisition_tool)

task_manager = FeatureTrackingTaskManager(
    llm_config=OpenAIConfig(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    image_acquisition_tool=acquisition_tool,
    image_registration_tool=registration_tool,
    message_db_path="messages.db",
)

task_manager.run_feature_tracking(
    reference_image_path="reference.png",
    initial_position=(0.0, 0.0),
    initial_fov_size=(64.0, 64.0),
    y_range=(-50.0, 50.0),
    x_range=(-50.0, 50.0),
)
```
