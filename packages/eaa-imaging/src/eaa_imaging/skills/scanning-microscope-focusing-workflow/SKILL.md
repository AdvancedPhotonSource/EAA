---
name: scanning-microscope-focusing-workflow
description: Procedure and instructions for focusing a scanning microscope.
---

## Overview

This document describes the procedure that should be followed to adjust the optics
parameters of a scanning microscope to reach the best focus. Often, the optics
parameters to be adjusted are the positions of focusing devices, such as a zone
plate or a lens. The evaluation of the focus is based on the line scan of a thin
feature, such as a thin line or a thin spot. The line scan of such features should
show a peak, and the FWHM of the peak indicates the quality of the focus. Better
focus is associated with smaller FWHM. The goal is to adjust the optics parameters
so that the FWHM is minimized.

## Tools

To perform the focusing task, you will need the following tools:

- `acquire_image`: Acquire an image of the region of interest.
- `acquire_line_scan`: Perform a line scan across the region of interest.
- `set_parameters`: Adjust the optics parameters.
- `get_current_parameters`: Get the current optics parameters.

If any of these tools are missing, notify the user.

## Procedure

Overall, the procedure consists of the following steps:

1. First, ask the user to confirm the location, size and other acquisition parameters
   (such as the scan step size) for acquiring a 2D image in a region of interest.
   Check the input arguments of the image acquisition tool to see what information
   is required. For spatial quantities like position and size, pay special attention
   to whether each value is in x or y direction. If unclear, ask the user to clarify.
   Additionally, ask the user for a recommended range to search for the optimal
   optics parameters.
   
2. Perform a 2D scan of the region of interest using the "acquire_image" tool. After
   acquiring the image, ask the user to provide you with the parameters for the line
   scan tool in order to perform a line scan across the thin feature to be used as
   landmark for evaluating the focus. Then run the line scan tool with the provided
   information.

3. The line scan tool will return a figure containing the following:
   - A line profile plot along the scan line and a Gaussian fit to the line profile.
     The FWHM of the Gaussian fit will be shown.
   - The path of the line scan you just did plotted in the last acquired 2D image
     (if an 2D image has been acquired previously).
   - The path of the first line scan plotted in the first acquired 2D image
     (if the current acquisition is not the first). This plot serves as the reference
     for the relative position of the scan line to the thin feature.

   The goal of a successful line scan is to have a scan line that
   - crosses the same landmark feature at the same relative location as the first (reference)
     line scan (Definition of "relative location": The image and the axis
     tick labels may undergo a global offset, but as long as the location of the scan line
     relative to other features in the image is the same, it is good).
   - crosses the landmark feature at the center of the scan line.

   3.1. Possible exceptions and handling: 
   You will see the last acquired 2D image with the scan line marked on it next to the
   line plot. The scan line should cross a thin feature, in most cases a thin line.
   The scan is considered abnormal if any of the following scenarios occurs:
   - The intersection of the scan line and the thin feature is very close to the
     edge of the scan line (which would result in an incomplete peak in the line plot).
   - The scan line almost overlaps with another feature in the image other than the
     landmark feature.
   - The location of the intersection relative to other features in the image 
     is very different from the last line scan. Note that the scan area of the 2D
     image itself may have drifted, so don't use the positions read from the axis ticks
     to judge whether the location of the intersection is abnormal. Instead, use the
     position relative to other features in the image to judge that.
   If any of these happens, discard the current line scan result, then adjust the positions
   and scan again. See the Examples section for illustrations of these abnormal cases
   and the position adjustments that should be applied.

4. Adjust the optics parameters using the parameter setting tool. You can confirm the current
   values of the parameters using the "get_current_parameters" tool. Use the 
   "set_parameters" tool to set the parameters. The stride of the parameter change
   shouldn't be too large. Normally, it should be within 1/10 or 1/20 of the user-recommended
   search range.

5. Acquire an image of the region using the image acquisition tool again using the same
   arguments as before. Due to the change of optics parameters, the image acquired
   may have drifted compared to the last one you saw, but most of the features in the
   previous image should still be visible in the new image.

   5.1. Possible exceptions and handling: if the image acquired has completely no common
      features with the previous image (NOT including periodic features such as repeating grid lines),
      it means that the optics parameters have been adjusted too much. In this case, you should
      adjust the optics parameters back to the previous values, then try again with a smaller stride.

6. Since the image has drifted, you need to adjust the positions where the next line scan is
   run. If you have an image registration tool, use it to register the new image with the previous image.
   When calling the tool, pass the explicit `.npy` paths for the current and previous images.
   To apply the registration result to the
   subsequent line scan, *add* the offset to the positions used for the previous line scan.
   Run a line scan with the updated positions. You will see the new line scan plot
   and the FWHM of the Gaussian fit. If you see any exceptions listed in 3, fix it and
   rerun line scan in the same way. Also, note down, and explicitly report the delta of the positions used for this
   line scan compared to the previous one. You will need this information to adjust the positions
   of subsequent operations.

7. Once you get the new FWHM, compare it with the trend of past FWHM values to determine
   the next optics parameter adjustment. Keep in mind that the goal is to minimize the
   FWHM, so you should adjust the optics parameters in the direction that makes the FWHM
   smaller. Make the parameter adjustment.

8. Repeat the process from step 5 (acquire 2D image). However, when acquiring this new
   image, you should add the delta of the positions found in step 6 to the last 2D scan
   positions. This ensures that the previous drifts are taken into account.

9. When the FWHM is minimized, conclude the process and report the history of optics parameters
   and the corresponding FWHM values.

## Criteria for Completion

You should reduce the stride of parameter adjustment when you are close to the optimal
focus. A recommended strategy is to move to smaller FWHM with a constant stride until
you overshoot (the FWHM starts to rise again), then turn around and conduct finer search
with a smaller stride. For example:

| zone plate z | FWHM |
| --- | --- |
| -190 | 5.2 |
| -191 | 3.8 |
| -192 | 2.3 |
| -193 | 1.5 |
| -194 | 2.4 |
| -193.5 | 1.8 |
| -192.5 | 1.6 |
| -192.2 | 1.2 |

The turning point occurs at -194, where the FWHM starts to increase again. This indicates
the optimum likely lies between -192 and -194. The operator then conducts finer search
within this interval at a finer resolution of 0.5, then reduced the stride further to
locate the optimum more precisely. Usually, you can conclude the process when you have
confidently found a minimum with a stride that is about 1/100 of the user-recommended
search range. 

However, bare in mind that the measurement and the FWHM values can be noisy. If you are
already at small stride size but find the FWHM values no longer have a clear trend,
it is also a good point to stop.

## Other Notes

- Tell the user your rationale behind each call to the data acquisition and parameter setting tools.
  When setting parameters, explain why you are setting them to those values. When running line scan
  or image acquisition, explain why you are doing them at the chosen positions.
- Be autonomous. Do not ask the user for input unless you miss critical information or
  you don't what to do.

## Examples

<message_break>

### Example of exceptions and handling in line scan (case 1)

Exception: intersection of landmark thin feature and scan line is away from the scan line’s center.

Fix: redo line scan, change x scan range to (150, 250) so that the thin vertical feature crosses the center of the scan line.
Also move it down to y = 180 so that the relative position of the scan line in the reference scan is restored.

![](images/focusing_line_scan_exceptions_example_1.png)

<message_break>

### Example of exceptions and handling in line scan (case 2)

Exception: scan line overlaps with another feature in the image.

Fix: redo line scan at y = 200 to move away from the overlapping feature.

![](images/focusing_line_scan_exceptions_example_2.png)

<message_break>

### Example of exceptions and handling in line scan (case 3)

Exception: the intersection between scan line and landmark feature drifts a lot relative to other features in the image compared to the reference scan.

Fix: redo line scan at x from 120, 220 and y = 250 to restore the relative location of the scan line in the reference scan. 

![](images/focusing_line_scan_exceptions_example_3.png)

<message_break>

### Example of exceptions and handling in line scan (case 4)

Exception: the scan line does not properly cross the landmark feature. 

Fix: redo line scan at x from 200, 300 and y = 150 so that the scan line crosses the isolated thin line
and the relative position of the scan line in the reference scan is restored.

![](images/focusing_line_scan_exceptions_example_4.png)

<message_break>

### Example of exceptions and handling in line scan (case 5)

Exception: the scan line is crossing an undesired feature (the short, thick, vertical block). 

Fix: redo the line scan at x from 20 to 120 and y = 100 to restore the relative position of the scan line in the reference scan.
If you have attempted this for several times but still can't get a good line scan, change the parameter to
a value closer the the previous one, then redo a 2D acquisition, and try again with line scan. This should
result in smaller image drift, making it easier to find a good line scan position.

![](images/focusing_line_scan_exceptions_example_5.png)

<message_break>

### Example of exceptions and handling in line scan (case 6)

Exception: the scan line is not at the same relative location compared to the reference scan. 

Fix: redo the line scan at x from 80 to 180 and y = 120 to restore the relative position of the scan line in the reference scan.

![](images/focusing_line_scan_exceptions_example_6.png)

<message_break>

## Conclusion

This concludes the instructions for the focusing workflow. Follow the given instructions,
and refer to the examples of exception recognition and handling to guide you through the process.
