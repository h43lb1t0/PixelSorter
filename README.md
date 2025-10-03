# PixelSorter

powerful Python application for glitch art and creative image manipulation through advanced pixel sorting. Sort images based on criteria like brightness, hue, or even isolate specific objects using YOLOv8 segmentation.

## Features
- Sort pixels by brightness, hue, saturation, etc.
- Choose different directions for sorting: horizontal (left to right, right to left), vertical (top to bottom, bottom to top).
- YOLOv8 object segmentation to isolate and sort specific objects within an image. [*](#custom-yolo-object-segmentation)
- Use any gray-scale image as a mask to define areas for pixel sorting.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/h43lb1t0/PixelSorter
    cd PixelSorter
    ```
2. Install the required dependencies:
(Install [UV](https://docs.astral.sh/uv/getting-started/installation/))
   ```bash
   uv sync
    ```

## Usage

All parameters are explained [here](#pixel-sorting-options).
The masking options are explained [here](#options).
### UI
Run the UI application:
```bash
python UI.py
```

### Command Line Interface (CLI)
```bash
usage: cli.py [-h] -i INPUT [-o OUTPUT] [-mi MASK_IMAGE] [-smo] [-yo]
              [-m MODEL] [-c CONF] [-bi BLUR_INCLUDE] [-be BLUR_EXTEND]
              [-ws {BACKGROUND,FOREGROUND,ALL}]
              [-sb {hue,saturation,brightness,lightness,luminance,color,red,green,blue,alpha,warmth,distance_center,distance_edges}]
              [-sd {ROW_LEFT_TO_RIGHT,ROW_RIGHT_TO_LEFT,COLUMN_TOP_TO_BOTTOM,COLUMN_BOTTOM_TO_TOP,SPIRALE_INWARD,SPIRALE_OUTWARD}]
              [-p]

A powerful pixel sorting tool with advanced masking capabilities.

options:
  -h, --help            show this help message and exit

Input/Output:
  -i, --input INPUT     Path to the input image or directory.
  -o, --output OUTPUT   Directory to save the sorted image(s). (Default: out/)

Masking Options:
  -mi, --mask-image MASK_IMAGE
                        Path to a custom black-and-white mask image.
                        This will override any YOLO-generated mask.
  -smo, --save-mask-only
                        Generate and save only the mask without sorting pixels.
                        Requires the --yolo flag.

YOLO Segmentation Options:
  -yo, --yolo           Use a YOLO model to automatically generate a mask from detected objects.
  -m, --model MODEL     The YOLO segmentation model to use. (Default: yolov8n-seg.pt)
  -c, --conf CONF       Confidence threshold for YOLO object detection (0.0 to 1.0). (Default: 0.35)
  -bi, --blur-include BLUR_INCLUDE
                        Blur include threshold for mask. (Default: 0.5)
  -be, --blur-extend BLUR_EXTEND
                        Blur extend threshold for mask. (Default: 0.7)
  -ws, --what-to-sort {BACKGROUND,FOREGROUND,ALL}
                        Specifies which part of the mask to sort. (Default: BACKGROUND)

Sorting Options:
  -sb, --sort-by {hue,saturation,brightness,lightness,luminance,color,red,green,blue,alpha,warmth,distance_center,distance_edges}
                        Method for sorting pixels. (Default: hue)
  -sd, --sort-direction {ROW_LEFT_TO_RIGHT,ROW_RIGHT_TO_LEFT,COLUMN_TOP_TO_BOTTOM,COLUMN_BOTTOM_TO_TOP,SPIRALE_INWARD,SPIRALE_OUTWARD}
                        The direction or pattern for the sorting algorithm. (Default: ROW_LEFT_TO_RIGHT)
  -p, --use-perlin      Introduce Perlin noise to randomize the sorting process.
```

### Code

Open the `main.py` file and modify the parameters as needed:
- `input_image_path`: Path to the input image.

#### Masks

If you do not want to use YOLOv8 object segmentation, comment out or remove the following lines:
```python
mask = YoloSegmentation(image).get_mask(
    what_to_sort=WhatToSort.BACKGROUND,
    conf=.35,
    blur_include=.5,
    blur_extend=.7
)
```
and set `mask = None`. in 
```python
sorted_image = PixelSorter(image).sort_pixels(
    sort_by=SortBy.luminance(),
    direction=SortDirection.COLUMN_BOTTOM_TO_TOP,
    mask=mask,
    use_perlin=False
)
```
### Options:
- `WhatToSort`: Choose whether to sort the background or the detected object.
- `conf`: Confidence threshold for object detection (between 0 and 1).
- `blur_include`: Blur threshold for including pixels in the mask (between 0 and 1).
- `blur_extend`: How much of the mask content should be extended outwars during bluring (between 0 and 1).

#### Custom YoLO Object Segmentation
If you want to use a model for segmenting persons I recommend downloading the model [yolo12l-person-seg-extended.pt](https://huggingface.co/RyanJames/yolo12l-person-seg/blob/main/yolo12l-person-seg-extended.pt) and set the model path in the `YoloSegmentation` class:
```python
mask = YoloSegmentation(image, model_path="path/to/yolo12l-person-seg-extended.pt").get_mask(
    what_to_sort=WhatToSort.BACKGROUND,
    conf=.35,
    blur_include=.5,
    blur_extend=.7
)
```

### Using a Custom Mask
To use a custom gray-scale image as a mask, load the mask image and pass it to the `sort_pixels` method:
```python
mask = Image.load_image("path/to/mask.png")
sorted_image = PixelSorter(image).sort_pixels(
    sort_by=SortBy.luminance(),
    direction=SortDirection.COLUMN_BOTTOM_TO_TOP,
    mask=mask,
    use_perlin=False
)
```

## Pixel Sorting Options
### SortBy
- `SortBy.hue()`: Sort by hue.
- `SortBy.saturation()`: Sort by saturation.
- `SortBy.brightness()`: Sort by brightness.
- `SortBy.lightness()`: Sort by lightness.
- `SortBy.luminance()`: Sort by luminance.
- `SortBy.color()`: Sort by color (RGB).
- `SortBy.red()`: Sort by red channel.
- `SortBy.green()`: Sort by green channel.
- `SortBy.blue()`: Sort by blue channel.
- `SortBy.alpha()`: Sort by alpha channel (Only works if the image has an alpha channel).
- `SortBy.warmth()`: Sort by warmth (red-yellow colors first).
- `SortBy.distance_center()`: Sort by distance to the center of the image.
- `SortBy.distance_edges()`: Sort by distance to the nearest edge of the image.

### SortDirection
- `SortDirection.ROW_LEFT_TO_RIGHT`: Sort rows from left to right.
- `SortDirection.ROW_RIGHT_TO_LEFT`: Sort rows from right to left.
- `SortDirection.COLUMN_TOP_TO_BOTTOM`: Sort columns from top to bottom.
- `SortDirection.COLUMN_BOTTOM_TO_TOP`: Sort columns from bottom to top.
- `SortDirection.SPIRALE_INWARD`: Sort in a spiral pattern inward from the edges to the center.
- `SortDirection.SPIRALE_OUTWARD`: Sort in a spiral pattern outward from the center to the edges.
(Spiral sorting may not work perfectly and especially with masks the results can be unpredictable and unsatisfactory.)

### Additional Options
- `mask`: A gray-scale image defining areas to sort. If `None`, the entire image is sorted.
- `use_perlin`: If `True`, applies Perlin noise to the sorting process for a more organic effect.


