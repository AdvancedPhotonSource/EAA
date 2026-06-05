"""All extents are expected to be given in pixel units. If the scan positions
are given in physical units, convert them to pixel units. You can infer the
pixel size using the physical extent of the image and the number of pixels
in each dimension which can be obtained from the raw data arrays.
"""


#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt


def parse_extent(values):
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Extent must be: start_y end_y start_x end_x")
    return tuple(map(int, values))


def load_2d_tiff(path):
    arr = tifffile.imread(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a monochannel 2D TIFF, got shape {arr.shape}")
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-tiff", required=True)
    parser.add_argument("--buffer-tiff-extent", nargs=4, required=True, metavar=("START_Y", "END_Y", "START_X", "END_X"))
    parser.add_argument("--new-view-tiff", required=True)
    parser.add_argument("--new-tiff-extent", nargs=4, required=True, metavar=("START_Y", "END_Y", "START_X", "END_X"))
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()

    buffer_extent = parse_extent(args.buffer_tiff_extent)
    new_extent = parse_extent(args.new_tiff_extent)

    by0, by1, bx0, bx1 = buffer_extent
    ny0, ny1, nx0, nx1 = new_extent

    buffer = load_2d_tiff(args.buffer_tiff)
    new = load_2d_tiff(args.new_view_tiff)

    expected_new_shape = (ny1 - ny0, nx1 - nx0)
    if new.shape != expected_new_shape:
        raise ValueError(
            f"New TIFF shape {new.shape} does not match its extent-derived shape "
            f"{expected_new_shape}"
        )

    expected_buffer_shape = (by1 - by0, bx1 - bx0)
    if buffer.shape != expected_buffer_shape:
        raise ValueError(
            f"Buffer TIFF shape {buffer.shape} does not match its extent-derived shape "
            f"{expected_buffer_shape}"
        )

    out_y0 = min(by0, ny0)
    out_y1 = max(by1, ny1)
    out_x0 = min(bx0, nx0)
    out_x1 = max(bx1, nx1)

    out_h = out_y1 - out_y0
    out_w = out_x1 - out_x0

    updated = np.zeros((out_h, out_w), dtype=np.result_type(buffer.dtype, new.dtype))

    buffer_dst_y0 = by0 - out_y0
    buffer_dst_y1 = buffer_dst_y0 + buffer.shape[0]
    buffer_dst_x0 = bx0 - out_x0
    buffer_dst_x1 = buffer_dst_x0 + buffer.shape[1]

    updated[buffer_dst_y0:buffer_dst_y1, buffer_dst_x0:buffer_dst_x1] = buffer

    new_dst_y0 = ny0 - out_y0
    new_dst_y1 = new_dst_y0 + new.shape[0]
    new_dst_x0 = nx0 - out_x0
    new_dst_x1 = new_dst_x0 + new.shape[1]

    updated[new_dst_y0:new_dst_y1, new_dst_x0:new_dst_x1] = new

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_tiff = save_dir / "updated_buffer.tiff"
    out_png = save_dir / "updated_buffer.png"

    tifffile.imwrite(out_tiff, updated)

    plt.figure(figsize=(8, 8))
    plt.imshow(
        updated,
        cmap="gray",
        origin="upper",
        extent=(out_x0, out_x1, out_y1, out_y0),
    )
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.title("Updated buffer")
    plt.colorbar(label="intensity")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"Saved TIFF: {out_tiff}")
    print(f"Saved PNG:  {out_png}")
    print(f"Updated extent: y=({out_y0}, {out_y1}), x=({out_x0}, {out_x1})")
    
    # If executing the script with the Python coding tool, print the output PNG path
    # at the end to allow the harness to display it.
    print(out_png)


if __name__ == "__main__":
    main()