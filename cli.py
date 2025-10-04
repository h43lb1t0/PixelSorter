import pathlib
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter

import cv2
from Enums import WhatToSort, SortDirection
from MaskHandler import YoloSegmentation
from PixelSorter import Image, PixelSorter, SortBy


def main():
    """Main function to parse arguments and run the pixel sorting process."""

    what_to_sort_options = [e.name for e in WhatToSort]
    sort_direction_options = [e.name for e in SortDirection]
    sort_by_options = SortBy.list_static_methods()

    # --- Argument Parser Setup ---
    parser = ArgumentParser(
        description="A powerful pixel sorting tool with advanced masking capabilities.",
        formatter_class=RawTextHelpFormatter
    )

    # --- Input/Output Group ---
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input image or directory."
    )
    io_group.add_argument(
        "-o", "--output",
        type=str,
        default="out/",
        help="Directory to save the sorted image(s). (Default: %(default)s)"
    )

    # --- Masking Group ---
    mask_group = parser.add_argument_group("Masking Options")
    mask_group.add_argument(
        "-mi", "--mask-image",
        type=str,
        help="Path to a custom black-and-white mask image.\nThis will override any YOLO-generated mask."
    )
    mask_group.add_argument(
        "-smo", "--save-mask-only",
        action="store_true",
        help="Generate and save only the mask without sorting pixels.\nRequires the --yolo flag."
    )

    # --- YOLO Segmentation Group ---
    yolo_group = parser.add_argument_group('YOLO Segmentation Options')
    yolo_group.add_argument(
        "-yo", "--yolo",
        action="store_true",
        help="Use a YOLO model to automatically generate a mask from detected objects."
    )
    yolo_group.add_argument(
        "-m", "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="The YOLO segmentation model to use. (Default: %(default)s)"
    )
    yolo_group.add_argument(
        "-c", "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for YOLO object detection (0.0 to 1.0). (Default: %(default)s)"
    )
    yolo_group.add_argument(
        "-bi", "--blur-include",
        type=float,
        default=0.5,
        help="Blur include threshold for mask. (Default: %(default)s)"
    )
    yolo_group.add_argument(
        "-be", "--blur-extend",
        type=float,
        default=0.7,
        help="Blur extend threshold for mask. (Default: %(default)s)"
    )
    yolo_group.add_argument(
        "-ws", "--what-to-sort",
        type=str,
        choices=what_to_sort_options,
        default=what_to_sort_options[0],
        help="Specifies which part of the mask to sort. (Default: %(default)s)"
    )

    # --- Sorting Group ---
    sort_group = parser.add_argument_group("Sorting Options")
    sort_group.add_argument(
        "-sb", "--sort-by",
        type=str,
        choices=sort_by_options,
        default=sort_by_options[0],
        help="Method for sorting pixels. (Default: %(default)s)"
    )
    sort_group.add_argument(
        "-sd", "--sort-direction",
        type=str,
        choices=sort_direction_options,
        default=sort_direction_options[0],
        help="The direction or pattern for the sorting algorithm. (Default: %(default)s)"
    )
    sort_group.add_argument(
        "-p", "--use-perlin",
        action="store_true",
        help="Introduce Perlin noise to randomize the sorting process."
    )

    args: Namespace = parser.parse_args()

    # --- 1. Post-parsing Argument Validation ---
    # Check if any YOLO-specific argument was changed from its default without --yolo being set.
    yolo_args_used = any([
        args.model != "yolov8n-seg.pt",
        args.conf != 0.35,
        args.blur_include != 0.5,
        args.blur_extend != 0.7,
        args.what_to_sort != what_to_sort_options[0]
    ])

    if yolo_args_used and not args.yolo:
        parser.error("Arguments like --model, --conf, etc., require the --yolo flag to be set.")

    if args.save_mask_only and not args.yolo:
        parser.error("--save-mask-only requires the --yolo flag to generate a mask.")

    if args.mask_image and args.yolo:
        print("Warning: --mask-image overrides --yolo. The YOLO model will not be used.")

    # --- 2. Setup Paths and Find Images ---
    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        return

    image_paths = []
    if input_path.is_file():
        image_paths.append(input_path)
    elif input_path.is_dir():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
            image_paths.extend(input_path.glob(ext))

    if not image_paths:
        print(f"No compatible images found in '{input_path}'.")
        return

    # --- 3. Instantiate YOLO Model ONCE (Performance Improvement) ---

    # --- 4. Main Processing Loop ---
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")
        suffix = img_path.suffix
        stem = img_path.stem
        img_path = str(img_path)
        image = Image.load_image(img_path)
        mask = None

        # --- Mask Generation ---
        if args.mask_image:
            print(f"-> Using custom mask: {args.mask_image}")
            mask_path = pathlib.Path(args.mask_image)
            if mask_path.exists():
                mask = Image.load_image(str(mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                print(f"Warning: Custom mask '{mask_path}' not found. Sorting entire image.")

        elif args.yolo:
            print("-> Generating YOLO mask...")
            mask = YoloSegmentation(image, model_path=args.model).get_mask(
                what_to_sort=WhatToSort[args.what_to_sort],
                conf=args.conf,
                blur_include=args.blur_include,
                blur_extend=args.blur_extend
            )

            if args.save_mask_only:
                mask_output_path = output_dir / f"{stem}_mask{suffix}"
                Image.save_image(mask, mask_output_path)
                print(f"   Saved YOLO mask to '{mask_output_path}'")
                continue  # Skip to the next image

        # --- Pixel Sorting ---
        print("-> Sorting pixels...")
        sorter = PixelSorter(image)
        sorted_image = sorter.sort_pixels(
            sort_by=getattr(SortBy, args.sort_by)(),
            direction=SortDirection[args.sort_direction],
            mask=mask,
            use_perlin=args.use_perlin
        )

        # --- Saving Output ---
        output_path = output_dir / f"{stem}_sorted{suffix}"
        Image.save_image(sorted_image, output_path)
        print(f"   Saved sorted image to '{output_path}'")

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()