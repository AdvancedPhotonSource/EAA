You are given a tool that acquires an image of a sub-region of a sample at given location and with given size (the field of view, or FOV). Each time you call the tool, you will see the image acquired. Use this tool to find a subregion that contains the following feature: {FEATURE_DESCRIPTION}. The feature should be centered in the field of view. Each time you see an acquired image, check if it is in the FOV; if not, move the FOV until you find it.

Here are your detailed instructions:
- At the beginning, use an FOV size of {FOV_SIZE} (height, width). You can change the FOV size during the process to see a larger area, but go back to this size when you find the feature and acquire a final image of it.
- Start from position (y={Y_MIN}, x={X_MIN}), and gradually move the FOV to find the feature. Positions should stay in the range of y={Y_MIN} to {Y_MAX} and x={X_MIN} to {X_MAX}.
- Use a regular grid search pattern at the beginning. Use a step size of {STEP_SIZE_Y} in the y direction and {STEP_SIZE_X} in the x direction. When you see the feature, you can move the FOV more arbitrarily to make it better centered.
- When you find the feature, adjust the positions of the FOV to make the feature centered in the FOV. If the feature is off to the left, move the FOV to the left; if the feature is off to the top, move the FOV to the top.
- Do not acquire images at the same or close location over and over again. If you find yourself calling the tool repeatedly at close locations, stop the process.
- When you find the feature of interest, report the coordinates of the FOV.
- Explain every tool call you make.
- When calling tools, make only one call at a time. Do not make another call before getting the response of a previous one.
- When you finish the search or need user response, say "TERMINATE".
