---
name: roi-search-workflow
description: Procedure for searching for a region of interest in a sample using microscope image acquisition tools during general chat.
---

## Overview

This skill describes how to search for a region of interest (ROI) in a sample
using a microscope image acquisition workflow. The goal is to find a field of
view (FOV) that contains a requested feature and then center that feature in the
FOV.

This skill is intended for general chat usage. It should not assume that the
required tools or inputs are already available.

## Required Tools

To perform ROI search effectively, you need:

- `acquire_image`: Acquire an image at a specified location and with a specified
  field of view.

If this tool is unavailable, notify the user that the ROI-search workflow cannot
be executed as described and explain what capability is missing.

## Required Inputs

Before starting the workflow, make sure you have the following information:

- A description of the feature to search for. This may optionally include a
  reference image if the user provides one.
- The allowed search range in y and x.
- The initial FOV size.
- The initial step size in y and x.

If any of this information is missing, ask the user for it before proceeding.
Check the tool schema as well, because the exact acquisition arguments may vary
across instruments.

## Procedure

Use the following workflow:

1. Confirm the feature being searched for and the acquisition settings needed
   to begin the search. In particular, confirm:
   - the feature description,
   - the starting search position,
   - the allowed y and x ranges,
   - the initial FOV size,
   - and the initial step sizes.

2. Start from one corner of the allowed search range and perform image
   acquisitions following a snake pattern with constant x and y strides. Use
   the user-provided or agreed-upon FOV size and step size.

   Example snake pattern:

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

3. After each acquired image, inspect whether the feature appears in the FOV.
   If not, continue the search systematically. At the beginning, prefer a
   regular grid-search pattern rather than arbitrary movement.

4. When the feature begins to appear in the image, refine the search more
   flexibly. Move the FOV in the direction needed to bring the feature toward
   the center. For example:
   - if the feature is to the left, move the FOV left;
   - if the feature is toward the top, move the FOV up.

5. You may temporarily increase the FOV size to inspect a larger area if that
   helps reacquire or localize the feature, but once the feature is found, return
   to the target FOV size and acquire a final centered image.

6. Avoid repeatedly acquiring images at the same or very similar locations. If
   you find yourself sampling nearly the same location over and over, stop and
   reconsider the search strategy instead of continuing blindly.

7. Once the feature of interest is found and centered, report the final FOV
   coordinates clearly to the user.

## Operating Guidance

- Explain the rationale for each acquisition call.
- Make only one tool call at a time.
- Respect the user-provided search bounds.
- When writing coordinates, explicitly label axes and use `(y = ..., x = ...)`
  format instead of listing unlabeled numbers.
- If the search cannot continue safely or productively because the feature is
  not identifiable, the bounds are too restrictive, or the required tool is not
  available, notify the user clearly.

## Completion

The workflow is complete when you have found the requested feature and centered
it in the FOV, then reported the final coordinates to the user.
