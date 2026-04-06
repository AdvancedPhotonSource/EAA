You will adjust the focus of a scanning microscope by adjusting the parameters of its optics. The focusing quality can be evaluated by performing a line scan across a thin feature and observing the FWHM of its Gaussian fit. The smaller the FWHM, the sharper the image. But each time you adjust the focus, the image may drift due to the change of the optics. You will need to perform a 2D scan prior to the line scan to locate the feature that is line-scanned.
{REFERENCE_IMAGE_PROMPTS}
{FEATURE_TEXT_DESCRIPTION}

Follow the procedure below to focus the microscope:

1. First, perform a 2D scan of the region of interest using the `"acquire_image"` tool and the following arguments: {SUGGESTED_2D_SCAN_KWARGS}. The image should look similar to the reference image. Determine the coordinates of the line scan path across the feature, and use the `"scan_line"` tool to perform a line scan across the feature. {LINE_SCAN_STEP_SIZE_PROMPT}
2. The line scan tool will return a plot along the scan line. You should see a peak in the plot. A Gaussian fit will be included in the plot and the FWHM of the Gaussian fit will be shown.
3. Adjust the optics parameters using the parameter setting tool. The initial parameter values are {INITIAL_PARAMETERS}.
4. Acquire an image of the region using the image acquisition tool. Here are the suggested arguments: {SUGGESTED_2D_SCAN_KWARGS}. The image acquired may have drifted compared to the last one you saw, but you should still see the line-scanned feature there. If not, try adjusting the image acquisition tool's parameters to locate that feature. {REGISTRATION_PROMPT}
5. Once you find the line-scanned feature, perform a new line scan across it again. Due to the drift, the start/end points' coordinates may need to be changed. {LINE_SCAN_POSITIONING_PROMPT}
6. You will be presented with the new line scan plot and the FWHM of the Gaussian fit.
7. Compare the new FWHM with the last one. If it is smaller, you are on the right track. Keep adjusting the parameters in the same direction. Otherwise, adjust the parameters in the opposite direction.
8. Repeat the process from step 4.
9. When you find the FWHM is minimized, you are done. Add "TERMINATE" to your response to hand over control back to the user.

Important notes:

- Your line scan should cross only one line feature, and you should see exactly one peak in the line scan plot. If there is not one, or if the Gaussian fit looks bad, check your arguments to the line scan tool and run it again. Make sure your line scan strictly follows the marker in the reference image. Do not trust the FWHM value in the line plot if there is no peak, or if the peak is incomplete.
- The line scan plot should show a complete peak. If the peak is incomplete, adjust the line scan tool's arguments to make it complete.
- The minimal point of the FWHM is indicated by an inflection of the trend of the FWHM with regards to the optics parameters. For example, if the FWHM is 3 with a parameter value of 10, then 1 with a parameter value of 11, then 3 with a parameter value of 12, this means the optimal parameter value is around 11.
{PARAM_STEP_SIZE_PROMPT}
- When calling a tool, explain what you are doing.
- When making a tool call, only call one tool at a time. Do not call multiple tools in one response.
- Remember that when coordinates are given in (y, x) order, the first coordinate is the row index (vertical) and the second is the column index (horizontal). When you write coordinates in your response, do not just write two numbers; instead, explicitly specify y/x axis and write them as (y = <y>, x = <x>).

When you finish or when you need human input, add "TERMINATE" to your response.
