---
name: microscopy-optics-tuning-bo-task-manager
description: Combines Bayesian optimization with feature tracking to tune microscope optics.
---

# Microscopy Optics Tuning BO Task Manager

## Overview
`MicroscopyOpticsTuningBOTaskManager` blends Bayesian optimization with feature tracking to tune
microscope optics. It iteratively adjusts parameters, re-centers the feature, and evaluates image quality.

## Task Manager Interface
Provide acquisition, parameter-setting, and BO tools along with optional feature-tracking kwargs.
Call `run()` to execute the optimization loop.

See `references/api_reference.md` for full signatures.

## Example
```python
import numpy as np

from eaa.api.llm_config import OpenAIConfig
from eaa.task_manager.tuning.bo_mic_optics import MicroscopyOpticsTuningBOTaskManager
from eaa.tool.optimization import BayesianOptimizationTool
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
bo_tool = BayesianOptimizationTool(bounds=([0.0, 0.0], [1.0, 1.0]))

task_manager = MicroscopyOpticsTuningBOTaskManager(
    llm_config=OpenAIConfig(api_key="YOUR_API_KEY", model="gpt-4o-mini"),
    image_acquisition_tool=acquisition_tool,
    parameter_setting_tool=parameter_tool,
    bayesian_optimization_tool=bo_tool,
    image_acquisition_kwargs={"loc_x": 0.0, "loc_y": 0.0, "size_x": 64, "size_y": 64},
    session_db_path="messages.db",
)

task_manager.run(n_iterations=5)
```
