from pathlib import Path

code = r'#!/usr/bin/env python3'
"""
Create a grid on one image, or align two gridded images and compare them.

Install:
    python -m pip install Pillow

Single-image examples:
    python compare_grid.py photo.jpg --rows 6 --cols 10
    python compare_grid.py photo.jpg --rows 6 --cols 10 --crop 100,80,900,600
    python compare_grid.py photo.jpg --rows 6 --cols 10 --select-crop

Two-image comparison:
    python compare_grid.py original.jpg --compare drawing.jpg --rows 6 --cols 10
    python compare_grid.py original.jpg --compare drawing.jpg --rows 6 --cols 10 \\
        --crop 100,80,900,600 --crop2 30,20,600,400
    python compare_grid.py original.jpg --compare drawing.jpg --rows 6 --cols 10 \\
        --select-crop --select-crop2

In comparison mode:
  * Both selected areas receive the same rows x columns grid.
  * Image 2's selected area is resized to Image 1's selected-area dimensions.
  * Grid-aligned versions, a 50/50 overlay, a side-by-side image, and a
    difference heatmap are saved.
  * --output is an output filename prefix, e.g. results/compare.
    The program adds suffixes such as _overlay.png and _difference.png.

Crop coordinates are x,y,width,height in pixels from each original image.
The comparison is a pixel-level visual comparison, not semantic recognition.
For best results, use similarly framed images with matching perspective.

For Commercial User
  * print blank grided pdf (ruler only, pencil grids for erasibility and grid for croping)
  * store in db - system independent
  * resizable output grids
  * Art improvement log
  * Art journey log (tagged art improvement log)
  * story/portfolio publishing
  * Digital Art Gallery
  * Commission Request ( No platform charges)
  * Art examination module

For Free User/python script download/MAC/WINDOWS/LINUX installer/ Free Tier
  * grid marks on picture
  * same grid marks on standard blank pdf
  * compare reference and drawing with grid marks reference
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageColor, ImageChops, ImageEnhance
except ImportError:
    print("Install Pillow first: python -m pip install Pillow", file=sys.stderr)
    raise SystemExit(2)


def parse_color(value):
    try:
        if re.fullmatch(r"#[0-9a-fA-F]{3}", value):
            return tuple(int(ch * 2, 16) for ch in value[1:])
        return ImageColor.getrgb(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Use a color name (red, blue, white) or hex value (#RRGGBB)."
        )


def parse_crop(value):
    try:
        x, y, w, h = (int(part.strip()) for part in value.split(","))
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError
        return x, y, w, h
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Crop must be x,y,width,height; x/y >= 0 and width/height > 0."
        )


def select_crop(image_path):
    """Select a rectangle with OpenCV. This optional path is not needed for CLI crops."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "Mouse crop selection requires OpenCV. Install it with "
            "'python -m pip install opencv-python', or use --crop X,Y,W,H."
        )
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    print(f"Select area in {image_path}; press ENTER/SPACE to confirm, ESC to cancel.")
    x, y, w, h = cv2.selectROI("Select grid area", frame, False, False)
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise RuntimeError("No crop area selected.")
    return int(x), int(y), int(w), int(h)


def open_rgb(path):
    with Image.open(path) as source:
        # Convert palette, grayscale, CMYK, and transparency modes consistently.
        return source.convert("RGB")


def resolve_area(image, crop):
    image_w, image_h = image.size
    x, y, area_w, area_h = crop or (0, 0, image_w, image_h)
    if x + area_w > image_w or y + area_h > image_h:
        raise ValueError(
            f"Crop {crop} exceeds image dimensions {image_w}x{image_h}."
        )
    return x, y, area_w, area_h


def draw_grid(image, rows, cols, color, line_width, crop):
    x, y, area_w, area_h = resolve_area(image, crop)
    draw = ImageDraw.Draw(image)
    for col in range(cols + 1):
        px = round(x + col * area_w / cols)
        draw.line((px, y, px, y + area_h), fill=color, width=line_width)
    for row in range(rows + 1):
        py = round(y + row * area_h / rows)
        draw.line((x, py, x + area_w, py), fill=color, width=line_width)
    return (x, y, area_w, area_h)


def grid_crop(image, area, rows, cols, color, line_width):
    """Return the selected area with a grid, keeping the original area aspect ratio."""
    gridded = image.copy()
    draw_grid(gridded, rows, cols, color, line_width, area)
    return gridded.crop((area[0], area[1], area[0] + area[2], area[1] + area[3]))


def make_difference_outputs(image1, image2, prefix):
    """Save aligned inputs, overlay, side-by-side view, and a red/yellow difference map."""
    # Both images are already the same size and have gridlines.
    overlay = Image.blend(image1, image2, 0.5)

    side_by_side = Image.new("RGB", (image1.width * 2, image1.height))
    side_by_side.paste(image1, (0, 0))
    side_by_side.paste(image2, (image1.width, 0))

    difference = ImageChops.difference(image1, image2)
    # Increase contrast so small pixel differences are easier to see.
    gray_difference = ImageEnhance.Contrast(difference.convert("L")).enhance(3.0)
    heatmap = ImageOps_colorize(gray_difference)

    outputs = {
        "_image1_grid.png": image1,
        "_image2_aligned_grid.png": image2,
        "_overlay.png": overlay,
        "_side_by_side.png": side_by_side,
        "_difference.png": difference,
        "_heatmap.png": heatmap,
    }
    for suffix, result in outputs.items():
        result.save(str(prefix) + suffix)

    # Mean absolute pixel difference, 0 = identical and 255 = maximum difference.
    stats = difference.convert("L").getextrema()
    mean_difference = sum(difference.convert("L").getdata()) / (
        difference.width * difference.height
    )
    return outputs, stats, mean_difference


def ImageOps_colorize(gray_image):
    """Map low differences to black, medium to yellow, and high to red/white."""
    # Use a simple RGB lookup table to avoid another dependency.
    lut = []
    for value in range(256):
        if value < 128:
            # Black -> yellow
            ratio = value / 127
            lut.extend((int(255 * ratio), int(220 * ratio), 0))
        else:
            # Yellow -> red-white
            ratio = (value - 128) / 127
            lut.extend((255, int(220 * (1 - ratio)), int(255 * ratio)))
    return gray_image.convert("RGB").point(lut)

def generate_blank_a4_pdf(
    rows,
    cols,
    landscape=False,
    margin_mm=15,
    color="black",
    line_width=1,
    output_dir=None
):
    """
    Generate a blank A4 PDF with corner ruler indicators.
    """
    try:
        DPI = 300
        MM_TO_INCH = 1 / 25.4

        if landscape:
            page_width_mm = 297
            page_height_mm = 210
        else:
            page_width_mm = 210
            page_height_mm = 297

        page_width = int(page_width_mm * MM_TO_INCH * DPI)
        page_height = int(page_height_mm * MM_TO_INCH * DPI)

        margin = int(margin_mm * MM_TO_INCH * DPI)

        image = Image.new(
            "RGB",
            (page_width, page_height),
            "white"
        )

        draw = ImageDraw.Draw(image)

        left = margin
        top = margin
        right = page_width - margin
        bottom = page_height - margin

        corner_size = int(5 * MM_TO_INCH * DPI)
        tick_length = int(2.5 * MM_TO_INCH * DPI)
        major_tick_length = int(5 * MM_TO_INCH * DPI)

        # Corner indicators
        corners = [
            (left, top, 1, 1),
            (right, top, -1, 1),
            (left, bottom, 1, -1),
            (right, bottom, -1, -1),
        ]

        for x, y, dx, dy in corners:
            draw.line(
                [(x, y), (x + dx * corner_size, y)],
                fill=color,
                width=line_width
            )
            draw.line(
                [(x, y), (x, y + dy * corner_size)],
                fill=color,
                width=line_width
            )

        # Top and bottom ruler ticks
        for i in range(cols + 1):
            x = left + (right - left) * i / cols

            length = (
                major_tick_length
                if i % 5 == 0 or i == cols
                else tick_length
            )

            draw.line(
                [(x, top), (x, top - length)],
                fill=color,
                width=line_width
            )

            draw.line(
                [(x, bottom), (x, bottom + length)],
                fill=color,
                width=line_width
            )

            draw.text(
                (x - 10, top - major_tick_length - 35),
                str(i),
                fill=color
            )

        # Left and right ruler ticks
        for i in range(rows + 1):
            y = top + (bottom - top) * i / rows

            length = (
                major_tick_length
                if i % 5 == 0 or i == rows
                else tick_length
            )

            draw.line(
                [(left, y), (left - length, y)],
                fill=color,
                width=line_width
            )

            draw.line(
                [(right, y), (right + length, y)],
                fill=color,
                width=line_width
            )

            draw.text(
                (left - major_tick_length - 35, y - 10),
                str(i),
                fill=color
            )

        # Automatically generate output filename
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        orientation = "landscape" if landscape else "portrait"

        output_path = output_dir / (
            f"blank_A4_{orientation}_{rows}x{cols}_{timestamp}.pdf"
        )
        image.save(
            output_path,
            "PDF",
            resolution=DPI
        )

        print(f"Blank A4 PDF saved: {output_path}")
    except Exception as e:
        print(str(e))

def build_parser():
    parser = argparse.ArgumentParser(
        description="Draw a grid on an image or compare two images using matching grids."
    )
    parser.add_argument("image", help="First/input image filename")
    parser.add_argument("--rows", "-r", type=int, required=True, help="Number of grid rows")
    parser.add_argument("--cols", "-c", type=int, required=True, help="Number of grid columns")
    parser.add_argument("--color", type=parse_color, default=parse_color("#ffffff"),
                        help="Grid line color (default: #ffffff)")
    parser.add_argument("--line-width", type=int, default=1,
                        help="Grid line thickness in pixels (default: 1)")

    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument("--crop", type=parse_crop, metavar="X,Y,W,H",
                            help="Grid only this rectangle in the first image")
    crop_group.add_argument("--select-crop", action="store_true",
                            help="Select first image crop with mouse (requires OpenCV)")

    parser.add_argument("--create-blank-a4", action="store_true", help="Generate a blank A4 PDF with corner ruler indicators")
    parser.add_argument("--compare", metavar="SECOND_IMAGE",
                        help="Compare this second image against the first image")
    parser.add_argument("--crop2", type=parse_crop, metavar="X,Y,W,H",
                        help="Crop rectangle for second image (comparison mode)")
    parser.add_argument("--select-crop2", action="store_true",
                        help="Select second image crop with mouse (requires OpenCV)")
    parser.add_argument("-o", "--output",
                        help="Output path; in comparison mode this is a filename prefix")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.rows < 1 or args.cols < 1 or args.line_width < 1:
        parser.error("--rows, --cols and --line-width must be positive integers.")
    if args.crop2 and not args.compare:
        parser.error("--crop2 can only be used with --compare.")
    if args.select_crop2 and not args.compare:
        parser.error("--select-crop2 can only be used with --compare.")
    if args.crop2 and args.select_crop2:
        parser.error("Use either --crop2 or --select-crop2, not both.")

    image1_path = Path(args.image).expanduser()
    if not image1_path.is_file():
        parser.error(f"Image file not found: {image1_path}")

    image2_path = Path(args.compare).expanduser() if args.compare else None
    if image2_path and not image2_path.is_file():
        parser.error(f"Second image file not found: {image2_path}")

    try:
        crop1 = select_crop(image1_path) if args.select_crop else args.crop
        image1 = open_rgb(image1_path)
        area1 = resolve_area(image1, crop1)

        if not args.compare:
            grid_image = image1.copy()
            draw_grid(grid_image, args.rows, args.cols, args.color,
                      args.line_width, area1)
            output = (Path(args.output).expanduser() if args.output else
                      image1_path.with_name(image1_path.stem + "_grid.png"))
            output.parent.mkdir(parents=True, exist_ok=True)
            grid_image.save(output)
            print(f"Saved grid image: {output.resolve()}")
            print(f"Grid: {args.rows} rows x {args.cols} columns")
            print(f"Grid area: {area1}")
            return

        crop2 = select_crop(image2_path) if args.select_crop2 else args.crop2
        image2 = open_rgb(image2_path)
        area2 = resolve_area(image2, crop2)

        # Draw identical row/column counts inside each selected region.
        gridded1 = grid_crop(image1, area1, args.rows, args.cols,
                             args.color, args.line_width)
        gridded2 = grid_crop(image2, area2, args.rows, args.cols,
                             args.color, args.line_width)

        # Normalize Image 2 to Image 1's selected grid-area dimensions.
        # This makes each corresponding grid cell occupy the same pixel dimensions.
        target_size = gridded1.size
        aligned2 = gridded2.resize(target_size, Image.Resampling.LANCZOS)

        prefix = Path(args.output).expanduser() if args.output else image1_path.with_name(
            image1_path.stem + "_vs_" + image2_path.stem
        )
        if prefix.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            prefix = prefix.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)

        _, extrema, mean_difference = make_difference_outputs(
            gridded1, aligned2, prefix
        )

        print("Comparison complete.")
        print(f"Image 1 grid area: {area1}; original size: {image1.size}")
        print(f"Image 2 grid area: {area2}; original size: {image2.size}")
        print(f"Normalized comparison size: {target_size[0]} x {target_size[1]}")
        print(f"Grid: {args.rows} rows x {args.cols} columns")
        print(f"Mean absolute grayscale difference: {mean_difference:.2f}/255")
        print(f"Difference range: {extrema[0]} to {extrema[1]}")
        print(f"Output filename prefix: {prefix.resolve()}")
        print("Created file suffixes:\n _image1_grid.png,\n _image2_aligned_grid.png,\n _overlay.png,\n _side_by_side.png,\n _difference.png,\n _heatmap.png\n")

        if args.create_blank_a4:
            generate_blank_a4_pdf(
                rows=args.rows,
                cols=args.cols,
            )

    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
