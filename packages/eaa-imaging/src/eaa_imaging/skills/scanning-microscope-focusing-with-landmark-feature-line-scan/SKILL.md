---
name: scanning-microscope-focusing-with-landmark-feature-line-scan
description: Procedure and instructions for focusing a scanning microscope using the landmark feature line scan method.
---

## Overview

This document describes the procedure to focus a scanning microscope using the
landmark feature line scan method. Often, the optics
parameters to be adjusted are the positions of focusing devices, such as a zone
plate or a lens. The evaluation of the focus is based on the line scan of a thin
feature, such as a thin line or a thin spot. The line scan of such features should
show a peak, and the FWHM of the peak indicates the quality of the focus. Better
focus is associated with smaller FWHM. The goal is to adjust the optics parameters
so that the FWHM is minimized.

## Tools and information needed

To perform the focusing task, you will need the following tools:

- A tool allowing you to acquire a 2D image of the sample at given locations.
- A line-scan tool allowing you to acquire a 1D line scan at given locations.
- A tool that allows you to adjust the parameters (or motors) of the focusing optics.
- A registration tool that allows you to find the offset between two images,
  given the paths of their raw data files. This allows you to "track" the landmark
  feature if the image drifts after optics parameter adjustment.

You will also need the following information:

- The raw image file (NPY or TIFF) of the first 2D scan in the region of 
  interest (ROI) containing the landmark;
  you need it for later image registration in order to track the feature if 
  the image drifts after optics parameter adjustment. If it is not available,
  ask the user for the position to conduct a 2D scan to acquire that image.
- The position of the first line scan across the landmark feature. It might be
  unavailable until the first 2D scan is completed because the line scan coordinates
  need to be read from the image. If this is the case, ask the user to confirm the
  line scan positions after the first 2D scan is done.
- The initial parameter(s) of the focusing optics. It might be a scalar number such
  as the z position of a zone plate, or it can be a vector of multiple parameters.
- The range to adjust the optics parameters for searching for the optimal focus.
- (Optional) A recommended initial step size by which the optics parameters should be adjusted
  each time at the beginning.
- (Optional) The desired precision of the optimal parameters (e.g., 0.5 micron).

If any of these tools or non-optional information are missing, ask the user.

## Procedure

Overall, the procedure consists of the following steps:

1. Conduct a 2D scan of the ROI if it is not yet available. Otherwise, if the user
   gives you the raw image file of the ROI's 2D scan, check that image. After that,
   confirm the landmark feature and the positions for the line scan across it.
   
2. Perform a 1D line scan across the landmark feature.

3. The line scan tool will return a figure containing the following:
   - A line profile plot along the scan line and a Gaussian fit to the line profile.
     The FWHM of the Gaussian fit will be shown.
   - The scan path of the scan you just did, plotted in the last acquired 2D image
     (if an 2D image has been acquired previously).
   - The scan path of the first line scan, plotted in the first acquired 2D image
     (if the current acquisition is not the first). This plot serves as the reference
     for the relative position of the scan line to the thin feature.

   The goal of a successful line scan is to have a scan line that
   - crosses the same landmark feature at the same location *relative to other features*
     in the sample.
   - crosses the landmark feature at the center of the scan line.

4. Adjust the optics parameters using the parameter setting tool.

5. After changing the optics parameters, the sample position relative to the scanning
   probe might drift. To calibrate, acquire an image of the region using the
   image acquisition tool again using the same arguments as before. The features of the
   image may have a translational shift compared to the previous one, but most features
   should be present in both images.

6. Since the image has drifted, you need to adjust the positions where the next line scan is
   run. Figure out the offset using the image registration tool. Unless explicitly mentioned
   otherwise, the registration tool should accept the paths to the raw data files of a moving
   (sometimes also called "current" or "test") image and a reference image. Denote the
   returned offset `[dy, dx]` as `registration_offset`. Its convention is:

   ```
   shifted_current_image = roll(current_image, registration_offset)
   ```

   where `shifted_current_image` aligns with the reference image. With this convention,
   make sure that you use the current image as the moving image, and use the last 2D image
   as the reference image unless a different reference is explicitly intended. The acquisition
   tool should return you the paths to the raw data files in NPY or TIFF format.
   
7. Assuming everything follows the convention mentioned in step 6, define the 2D
   scan-position difference and the stage-position correction as follows:

   ```
   scan_position_difference = current_2d_scan_position - last_2d_scan_position
   stage_position_correction = scan_position_difference - registration_offset
   new_line_scan_position = previous_line_scan_position + stage_position_correction
   ```

   This reduces to simply subtracting `registration_offset` only when the current and last
   2D scans were acquired at the same position. Conduct another line scan at the
   corrected position. You should see that the scan line crosses the landmark feature
   at the same location relative to other features in the sample, despite that the
   absolute positions indicated in the axis ticks may differ.

8. Once you get the new FWHM from the line scan, compare it with the trend of past FWHM values to determine
   the next optics parameter adjustment. Keep in mind that the goal is to minimize the
   FWHM, so you should adjust the optics parameters in the direction that makes the FWHM
   smaller. Then use the parameter setting tool or motor moving tool to make the parameter adjustment.

8.1. Optionally, after getting the FWHM, you may use your Python tool to visualize the data
  points measured so far which helps you understand where you are in the search. If you
  decided to use Bayesian optimization (which is agian optional), you can also update the
  model and get new suggestion in this step. 

9. Repeat the process from step 5 (acquire 2D image). When acquiring the next
   image, update the last 2D scan position by the same `stage_position_correction`
   defined above:

   ```
   new_2d_scan_position = previous_2d_scan_position + stage_position_correction
   ```

   This ensures that the previous drift is taken into account before the next
   registration and line scan.

10. When the FWHM is minimized, conclude the process and report the history of optics parameters
    and the corresponding FWHM values.

When you finish, generate a plot of the parameter - FWHM data points (if the parameter
is 1D or 2D), and fit a quadratic function to it.

## Strategies of optimization

### Initial direction

The user may suggest to you the initial direction of moving the optics - for example, "we are currently
at z = -198, the right focus might be on the positive side". If the user does not suggest that, move around
the initial point slightly to figure out the direction along which the FWHM descends. Remember that
drift calibration should also be done while collecting these data points.

### Advanced search methods

Usually, if the parameter being optimized is a scalar, your own decision of adjusting
the parameters should be good enough. However, if the data points of paraemter - FWHM
pairs are exetremely noisy, or if the parametes are multivariate, consider the following
to help with your search:
- You may use Bayesian optimization to help with your search. Use a high-level BO package
  such as `botorch` directly if it is available in the runtime environment, otherwise
  create a local `uv` environment and install it. Use the measured data to fit the model,
  choose a suitable acquisition function (such as expected improvement), and find the
  optimum of the acquisition function to determine the next step. However, if BO suggests
  a new parameter that is very far away from the current values, clip the change to a
  smaller step length to prevent unmanageable drift of the sample images.
- Use fitting. You may use your Python tool to fit the data points (e.g., with a 
  quadratic function) to estimate the optimum.
  If you use the fit to estimate the optimum, make sure you constrain the step length
  like in the case of BO.

### Use visualization

Use your Python tool often to plot the data points measured so far to visualize the state
of the search. This helps you to understand the trends of the FWHM and also helps you
decide whether the current minimum is a noise-induced fluctuation or the true optimum.

### Handling noise

The FWHM values may be noisy. If you are searching with very fine step size and find a
lot of fluctuations in the data points, do not blindly choose the minimum FWHM. Instead,
fit a quadratic function and plot the data points and the fit to help you decide where
the true optimum is.

### Coase-to-fine search

Use a reasonable step size at the beginning to roughly locate the optimum, then conduct
finer search around the vicinity of the optimum to precisely locate it. For example,
when focusing a zone plate, you may first step the z-position of the zone plate at
1 mm until you find the optimum with a resolution of 1 mm. If the user wants a finer
resolution of 0.1 mm, search within the +/- 1mm window of the optimum with a step size
of 0.1 mm following that to locate the precise optimum. See the *Criteria for Completion*
section for more details. 

## Avoid early termination

Noise and imprecision in line scan positions can cause fluctuations in the FWHM.
Do not trust the immediate data point that appears like "overshooting past the
optimum". For example, if you see

| zone plate z | FWHM |
| --- | --- |
| -190 | 5.2 |
| -191 | 3.8 |
| -192 | 2.3 |
| -193 | 2.6 |

Do not conclude that the optimum is between -193 and -191 right away. Keep going in that
direction a little until you see a consistent increasing trend in the FWHM. When in doubt,
plot the data points to better visualize the state.

## Handling exceptions

1. The 2D image acquisition tool does not return the raw data path (not PNG)

You need the raw data file to use the image registration tool to accurately
decide how much the image and line scan positions should be shifted to track
the landmark feature. If the raw path is unavailable, you may try to directly
read out the position of the drifted line scan position from the axis ticks
of the new image if you are confident. Otherwise, notify the user.

2. The line scan path no longer crosses the landmark feature at the same relative location

This is usually because image registration failed to give an accurate result. 
Immediately revert to the parameter setting before the last adjustment, and
redo a line scan at the same position used at that time. Confirm the state is reverted
by checking if the scan line is restored to the same relative location as in
the reference image. Then, adjust the optics parameters again in the same direction
as you did last time, but with a smaller step size to reduce the drift, which
should make image registration easier.

The following images demonstrate a few cases where the line scan path goes "off".

![](images/focusing_line_scan_exceptions_example_1.png)

![](images/focusing_line_scan_exceptions_example_2.png)


## Other Notes

- Tell the user your rationale behind each call to the data acquisition and parameter setting tools.
  When setting parameters, explain why you are setting them to those values. When running line scan
  or image acquisition, explain why you are doing them at the chosen positions.
- You should perform the focusing as autonomous as possible. Unless major exceptions happen, do
  not request human intervention during the process. Before the focusing process concludes,
  avoid using the bash tool because it would ask for user approval and may cause the process to
  hang.
