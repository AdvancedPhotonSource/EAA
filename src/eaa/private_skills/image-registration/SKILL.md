---
name: image-registration
description: Instructions for estimating translational shift between two images.
---

You will be shown one image containing a pair of images side by side:
- Left: previous/reference image.
- Right: current image.

The images may have different noise level, sharpness and color scale, but
they should contain common structures and features. Use the common structures to
estimate the pure translational offset between them. Your offset must mean:
- If the image on the left is shifted by this offset, it aligns with the image on the right.
- Report shift in ij indexing: `<shift_y>, <shift_x>` (y first, x second).
- Positive y means down. Positive x means right.
- Report shift values as fractions of image size:
  - `shift_y` is a fraction of image height.
  - `shift_x` is a fraction of image width.

Example:
- If the left image should be shifted downward by 30% of image height and left by 10% of image width,
  report `0.3, 0.1`.

Before responding with the final shift:
- Use the `apply_and_view_offset` tool to verify your proposed offset.
- Provide `register_with="previous"`, `fractional_offset_y`, and `fractional_offset_x`.
- You will see two images plotted side-by-side: the left will be the reference/previous image shifted by the specified amount, and the right will be the current image. Check if they are aligned. Refine the offset if needed.
- You may call this tool multiple times until alignment is satisfactory.

Output format requirements:
- Return exactly one line in this exact format: `<shift_y>, <shift_x>`.
- Do not add any extra text.
- If the two images have no overlap and cannot be registered, return exactly: `nan, nan`.

If you can't get a good alignment after 5 attempts, return `nan, nan`.

Make NaN handling consistent: use lowercase `nan` exactly when registration is not possible.

Sometimes the input images might have been already reasonably aligned. In that case, just report `0, 0`.

<message_break>

## Examples

### Example 1

Correct fractional offset: -0.13, -0.17

![](images/registration_example_1.png)

<message_break>

### Example 2

Correct fractional offset: 0.1, -0.1

![](images/registration_example_2.png)

<message_break>

### Example 3

Correct fractional offset: 0.15, -0.18

![](images/registration_example_3.png)

<message_break>

### Example 4

Correct fractional offset: -0.03, -0.03

![](images/registration_example_4.png)

<message_break>

### Example 5

Correct fractional offset: 0, 0

![](images/registration_example_5.png)