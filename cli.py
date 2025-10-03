from MaskHandler import YoloSegmentation
from PixelSorter import PixelSorter, Image, SortBy
from Enums import WhatToSort, SortDirection
from argparse import ArgumentParser, Namespace
import pathlib


def main():

    what_to_sort_options = [arr for arr in WhatToSort.__members__.keys()]
    sort_direction_options = [arr for arr in SortDirection.__members__.keys()]
    sort_by_options = SortBy.list_static_methods()

    parser = ArgumentParser(description="Pixel sorting")
    yolo_mask_group = parser.add_argument_group("Mask options")
    sort_group = parser.add_argument_group("Sorting options")
    
    #Input Image/Directory
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image or directory (for batch processing)")

    parser.add_argument("-o", "--output", type=str, default="out/", help="Output directory. Default is 'out/'. The file name will be the same as the input image with '_sorted' appended before the file extension.")

    #Mask options
    ## Use YOLO segmentation
    yolo_mask_group.add_argument("-yo", "--yolo", action="store_true", help="Use YOLO segmentation to create a mask")
    yolo_mask_group.add_argument("-m", "--model", type=str, default="yolov8n-seg.pt", help="YOLO model to use. Default is 'yolov8n-seg.pt'")
    yolo_mask_group.add_argument("-c", "--conf", type=float, default=0.35, help="Confidence threshold for YOLO model. Default is 0.35")
    yolo_mask_group.add_argument("-bi", "--blur-include", type=float, default=0.5, help="Blur include threshold for mask. Default is 0.5")
    yolo_mask_group.add_argument("-be", "--blur-extend", type=float, default=0.7, help="Blur extend threshold for mask. Default is 0.7")
    yolo_mask_group.add_argument("-ws", "--what-to-sort", type=str, choices=what_to_sort_options, default=what_to_sort_options[0], help=f"What to sort: {what_to_sort_options}. Default is {what_to_sort_options[0]}")

    ## Use custom mask
    yolo_mask_group.add_argument("-mi", "--mask-image", type=str, help="Path to custom mask image. Overrides YOLO mask if both are provided.")

    #Sorting options
    sort_group.add_argument("-sb", "--sort-by", type=str, choices=sort_by_options, default=sort_by_options[0], help=f"Method to sort by: {sort_by_options}. Default is {sort_by_options[0]}")
    sort_group.add_argument("-sd", "--sort-direction", type=str, choices=sort_direction_options, default=sort_direction_options[0], help=f"Direction to sort: {sort_direction_options}. Default is {sort_direction_options[0]}")
    sort_group.add_argument("-p", "--use-perlin", action="store_true", help="Use Perlin noise for sorting")


    args: Namespace = parser.parse_args()
    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_yolo = args.yolo
    mask_image_path = args.mask_image
    what_to_sort = WhatToSort[args.what_to_sort]
    conf = args.conf
    blur_include = args.blur_include
    blur_extend = args.blur_extend
    sort_by = SortBy.from_string(args.sort_by)
    sort_direction = SortDirection[args.sort_direction]
    use_perlin = args.use_perlin

    if not input_path.exists():
        print(f"Input path '{input_path}' does not exist.")
        return
    
    input_images = []
    if input_path.is_file():
        input_images.append(input_path)
    elif input_path.is_dir():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
            input_images.extend(input_path.glob(ext))
    else:
        print(f"Input path '{input_path}' is neither a file nor a directory.")
        return
    if len(input_images) == 0:
        print(f"No images found in '{input_path}'.")
        return
    
    for img_path in input_images:
        if not args.mask_image and not use_yolo:
            sorted_image = PixelSorter(Image.load_image(img_path)).sort_pixels(
                sort_by=sort_by,
                direction=sort_direction,
                mask=None,
                use_perlin=use_perlin
            )
            Image.save_image(sorted_image, output_dir / f"{img_path.stem}_sorted{img_path.suffix}")
            print(f"Saved sorted image to '{output_dir / f'{img_path.stem}_sorted{img_path.suffix}'}'")
        
        elif mask_image_path:
            mask = Image.load_image(mask_image_path, as_mask=True)
            sorted_image = PixelSorter(Image.load_image(img_path)).sort_pixels(
                sort_by=sort_by,
                direction=sort_direction,
                mask=mask,
                use_perlin=use_perlin
            )
            Image.save_image(sorted_image, output_dir / f"{img_path.stem}_sorted{img_path.suffix}")
            print(f"Saved sorted image to '{output_dir / f'{img_path.stem}_sorted{img_path.suffix}'}'")

        elif use_yolo:
            image = Image.load_image(img_path)
            mask = YoloSegmentation(image, model_path=args.model).get_mask(
                what_to_sort=what_to_sort,
                conf=conf,
                blur_include=blur_include,
                blur_extend=blur_extend
            )
            sorted_image = PixelSorter(image).sort_pixels(
                sort_by=sort_by,
                direction=sort_direction,
                mask=mask,
                use_perlin=use_perlin
            )
            Image.save_image(sorted_image, output_dir / f"{img_path.stem}_sorted{img_path.suffix}")
            print(f"Saved sorted image to '{output_dir / f'{img_path.stem}_sorted{img_path.suffix}'}'")

        else:
            print("No valid mask option provided. Use --yolo or provide a custom mask image with --mask-image.")
            return




if __name__ == "__main__":
    main()