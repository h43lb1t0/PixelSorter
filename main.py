from MaskHandler import YoloSegmentation
from PixelSorter import PixelSorter, Image, SortBy
from Enums import WhatToSort, SortDirection

input__image_path = r"e:\Bilder\Photos-1-001\bearbeitet_DSC232116.jpg"
output_dir = "examples/out/"

input_image_name = input__image_path.split("/")[-1].split("\\")[-1].split(".")[0]
output_path = f"{output_dir}{input_image_name}_sorted.png"

image = Image.load_image(input__image_path)

mask = YoloSegmentation(image, model_path="yolo12l-person-seg-extended.pt").get_mask(
    what_to_sort=WhatToSort.FOREGROUND,
    conf=.35,
    blur_include=.7,
    blur_extend=.7
)

sorted_image = PixelSorter(image).sort_pixels(
    sort_by=SortBy.luminance(),
    direction=SortDirection.COLUMN_BOTTOM_TO_TOP,
    mask=mask,
    use_perlin=False
)

Image.save_image(sorted_image, output_path)