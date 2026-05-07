You are given a reference image of a field of view (FOV) of a microscope. After that reference image was collected, the imaging system drifted, so the FOV at the same location is now at a different location. Use the tool to acquire images at different places to find that feature again, and bring it back to the same location of the field of view as the reference image. Try zooming out the field of view, and move around to find the feature, then zoom back in. The new image you acquire might be blurrier than the reference.

- For your first attempt, start from the initial position of x = {INITIAL_X}, y = {INITIAL_Y}, and use an initial field of view size of {INITIAL_FOV_X} in x and {INITIAL_FOV_Y} in y.

- After acquiring the first image, try zooming out and moving the FOV around by calling the acquisition tool with different locations and sizes/step sizes until the acquired image has substantial overlap with the reference image.

- When the acquired image has substantial overlap with the reference, change the FOV size back to the initial size while keeping the location of the FOV the same as your latest acquisition.

- Adjust the final field of view so that the final image you see is as aligned with the reference image as possible. If you have an image registration tool, use it to perform the precise alignment. However, only use your tool if you see a large amount of overlap between the last acquired image and the reference, otherwise registration will not be accurate. Call the registration tool with explicit `.npy` image paths for the current image and reference image. The offset returned by the registration tool should be subtracted from the positions of the image acquisition tool. For example, if the last image is acquired at (y = 100, x = 100), and the registration tool returns an offset of (dy, dx), the next image should be acquired at (y = 100 - dy, x = 100 - dx).

- After the last acquired image is aligned with the reference, report the coordinates of the field of view and include "TERMINATE" in your response.

Other notes:

- Unless specifically noted, all coordinates are given in (y, x) order. When writing coordinates in your response, do not just write two numbers; instead, explicitly specify y/x axis and write them as (y = <y>, x = <x>).

- Always explain your actions when calling a tool.

- Make sure you only make one tool call at a time. Do not make multiple calls simultaneously.

- Do not acquire images at the initial location over and over again. You do not need to acquire baseline images for registration.

- When the acquired image looks well aligned with the reference, stop the process by adding "TERMINATE" to your response.
