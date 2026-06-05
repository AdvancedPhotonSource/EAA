---
name: roi-search-workflow
description: Procedure for searching for a region of interest in a sample using microscope image acquisition tools during general chat.
---

## Overview

This skill describes how to search for a region of interest (ROI) in a sample
using a microscope image acquisition workflow. The goal is to find a field of
view (FOV) that contains a requested feature and then center that feature in the
FOV.

The desired ROI either contains a specific feature of interest or has structure
that matches the user's description or reference images. Search can use various
strategies, ranging from a basic acquire-and-check approach to a more
sophisticated coarse-to-fine, multi-scale method. The image acquisition tool is
your core asset: it lets you acquire an image at a given location, optionally
with a specified sampling interval (scan step size) and FOV size. Use it to
collect images of the sample at different locations and with different
configurations to achieve the goal.

## Required Tools

To perform ROI search effectively, you need:

- `acquire_image`: Acquire an image at a specified location and with a specified
  field of view.

If this tool is unavailable, notify the user that the ROI-search workflow cannot
be executed as described and explain what capability is missing.

## Required Inputs

Before starting the workflow, make sure you have the following information:

- A description of the feature to search for, or a reference image of the desired
  feature provided by the user.
- User-suggested search strategy (optional).
- The allowed search range in x and y.
- The initial FOV size.
- The initial step size in x and y (unless the user explicitly asks you to avoid
  searching with fixed step sizes).
- The goal: does the final FOV need to be centered on the described feature or
  aligned with the reference image, or is it enough for the described or shown
  features to be visible in the FOV?

If any essential information is missing, ask the user for it before proceeding.
Check the tool schema as well, because the exact acquisition arguments may vary
across instruments. For example, some image acquisition tools may not take an
FOV size.

## Search Strategy

The most fundamental search strategy is an acquire-check-move loop:

1. Acquire an image at the corner of the search range.
2. Check whether the desired feature is in the image or whether the image
   matches the user-provided reference.
3. If not, move to the next location and repeat the acquisition and checking.

This strategy should be used if the user did not specify any strategy.

By default, move the FOV with constant x and y strides. If the user did not
specify a stride, use the FOV size so adjacent images are tightly connected. The
default search path in the sample space should be a snake pattern with constant
x and y strides. Here is an example snake pattern:

```text
Snake-pattern scan with constant x and y strides
(start at lower-left corner in this example)

y ^
   |
   |   row 4   o ---> o ---> o ---> o
   |                                 |
   |   row 3   o <--- o <--- o <--- o
   |   |
   |   row 2   o ---> o ---> o ---> o
   |                                 |
   |   row 1   o <--- o <--- o <--- o
   |   |
   |   row 0   o ---> o ---> o ---> o
   |
   +----------------------------------------> x

      x stride = constant horizontal spacing
      y stride = constant vertical spacing
```

The user may also suggest a search strategy. For example, one possible strategy
is to run a coarse or large-spaced scan across the entire sample or a large area
within it, then perform finer searches in areas that may contain the desired
feature. Use the image acquisition tool flexibly to follow the user's
instructions. If the image acquisition tool does not support the specified
strategy, notify the user and suggest falling back to the default
acquire-check-move snake pattern.

Use your Python coding tool creatively and flexibly to perform necessary
analyses on the fly or to visualize the search state. For example, at the end
of a coarse or random search pass, you can generate a large image with local
images placed at their respective locations. Use that overview of the covered
sample area to help determine where to scan next. Some analyses may require
access to raw image data stored on disk in formats such as NPY, TIFF, or HDF5.
The acquisition tool might return paths to these raw data files, but this is
not guaranteed.

## Tips

### Visualization

Use your Python coding tool to create visualizations that help you understand
the search state. For example, if you have access to the raw image data,
you can generate a script that creates a large image buffer, and place the
acquired images into that buffer at their respective locations, stitching
them together into a panorama of the searched area, and update the image
after each acquisition. This can help you see which areas have been covered and
identify promising regions to focus on in subsequent search passes.
See `update_and_view_panorama.py` in the skill directory for an example of how 
to do this.

## Finalization

When the feature begins to appear in the image, refine the search more
flexibly. If the user needs the final FOV to be centered on the desired feature,
adjust the acquisition location in the direction needed to bring the feature toward
the center. For example:
- if the feature is to the left, move the FOV left;
- if the feature is toward the top, move the FOV up.

Once the feature of interest is found and, if necessary, centered, report the
final FOV coordinates clearly to the user.

## Operating Guidance

- Explain the rationale for each acquisition call.
- Make only one tool call at a time.
- Respect the user-provided search bounds.
- When writing coordinates, explicitly label axes and use `(x = ..., y = ...)`
  format instead of listing unlabeled numbers.
- If the search cannot continue safely or productively because the feature is
  not identifiable, the bounds are too restrictive, or the required tool is not
  available, notify the user clearly.
- Avoid repeatedly acquiring images at the same or very similar locations. If
  you find yourself sampling nearly the same location over and over, stop and
  reconsider the search strategy instead of continuing blindly.
