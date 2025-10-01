from MaskHandler import YoloSegmentation
from PixelSorter import PixelSorter, Image, SortBy
from Enums import WhatToSort, SortDirection

input__image_path = "examples/alone-4480442.jpg"
output_path = "examples/out/out.png"

image = Image.load_image(input__image_path)

mask = YoloSegmentation(image).get_mask(
    what_to_sort=WhatToSort.BACKGROUND,
    conf=.4,
    blur_include=.7,
    blur_extend=.7
)

sorted_image = PixelSorter(image).sort_pixels(
    sort_by=SortBy.hue(),
    direction=SortDirection.COLUMN_BOTTOM_TO_TOP,
    mask=mask
)

Image.save_image(sorted_image, output_path)