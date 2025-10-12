# `SortBy` Methods Explained


Below is a detailed explanation of each sorting option, complete with a visual example (Sort direction: left to right) using the example image:
<img src="examples/20250413-_DSC5146.jpg" alt="original" height="250">

---

## Color Space & Perceptual Sorting

These methods sort pixels based on properties derived from color models like HSV and CIELAB, which often align more closely with human perception than raw RGB values.

### `brightness()`

Sorts pixels from darkest to lightest based on the **V**alue component of the **HSV** color model.

<img src="examples/sorted_by_option/brightness.jpg" alt="brightness" height="250">

### `hue()`

Sorts pixels based on their **H**ue, which represents the pure color (e.g., red, green, yellow). This often results in a rainbow-like gradient. It uses saturation and brightness as tie-breakers.

<img src="examples/sorted_by_option/hue.jpg" alt="hue" height="250">

### `saturation()`

Sorts pixels by their color intensity or "purity," from grayscale/muted colors to the most vibrant colors. This uses the **S**aturation component of the **HSV** color model.

<img src="examples/sorted_by_option/saturation.jpg" alt="saturation" height="250">

### `lightness()`

Sorts pixels by their perceptual lightness, using the **L\*** component from the **CIELAB** color space. This model is designed to be perceptually uniform, meaning changes in the value correspond closely to what the human eye perceives as a change in lightness.

<img src="examples/sorted_by_option/lightness.jpg" alt="lightness" height="250">

### `luminance()`

Sorts pixels based on their calculated luminance, which is a weighted sum of the R, G, and B values that approximates how bright the human eye perceives a color to be.

<img src="examples/sorted_by_option/luminance.jpg" alt="luminance" height="250">

---

## RGB Channel Sorting

These methods sort pixels based on the intensity of their individual Red, Green, and Blue channels or a specific combination thereof.

### Individual Channels: 

- `red()`
<img src="examples/sorted_by_option/red.jpg" alt="red" height="250">
- `green()`
<img src="examples/sorted_by_option/green.jpg" alt="green" height="250">
- `blue()`
<img src="examples/sorted_by_option/blue.jpg" alt="blue" height="250">

These methods sort pixels based on the intensity of a single color channel, from lowest (0) to highest (255).



### Channel Combinations: `rgb()`, `rbg()`, `bgr()`, etc.

These methods sort pixels by prioritizing the RGB channels in a specific order. For example, `bgr()` will first sort all pixels by their blue value. For any pixels with the same blue value, it will then sort them by their green value, and finally by their red value as a last tie-breaker.

| Function | Color Order Example |
| :--- | :--- |
| `rgb()` | Red -> Green -> Blue <br> <img src="examples/sorted_by_option/rgb.jpg" alt="rgb" height="250"> |
| `rbg()` | Red -> Blue -> Green <br> <img src="examples/sorted_by_option/rbg.jpg" alt="rbg" height="250"> |
| `gbr()` | Green -> Blue -> Red <br> <img src="examples/sorted_by_option/gbr.jpg" alt="gbr" height="250"> |
| `grb()` | Green -> Red -> Blue <br> <img src="examples/sorted_by_option/grb.jpg" alt="grb" height="250"> |
| `bgr()` | Blue -> Green -> Red <br> <img src="examples/sorted_by_option/bgr.jpg" alt="bgr" height="250"> |
| `brg()` | Blue -> Red -> Green <br> <img src="examples/sorted_by_option/brg.jpg" alt="brg" height="250"> |


### `alpha()`

Sorts pixels based on their transparency. This requires the input image to have an alpha channel (e.g., a PNG file). Pixels with lower alpha values (more transparent) will appear first, while fully opaque pixels will appear last.

---

## Other Color-Based Sorting

These methods use custom calculations based on pixel color values to achieve unique sorting effects.

### `warmth()`

Sorts pixels from "cool" (high in blue) to "warm" (high in red and green). The sorting key is calculated as `R + G - B`.

<img src="examples/sorted_by_option/warmth.jpg" alt="warmth" height="250">

### `chroma()`

Sorts pixels from low-chroma (grayscale, muted) to high-chroma (pure, vibrant). The key is the difference between the maximum and minimum RGB channel values for each pixel.

<img src="examples/sorted_by_option/chroma.jpg" alt="chroma" height="250">

### Grayscale Sorting
These methods create a grayscale-like effect by sorting pixels based on the **mean** (average) or **median** of their R, G, and B channel values.

- `mean()` 
<img src="examples/sorted_by_option/mean.jpg" alt="mean" height="250">
- `median()`
<img src="examples/sorted_by_option/median.jpg" alt="median" height="250">





### `contrast_to_neighbors()`

Sorts pixels based on how much their luminance contrasts with the average luminance of the entire slice being sorted. Pixels with luminance values far from the average (either much darker or much lighter) are placed after those closer to the average.

<img src="examples/sorted_by_option/contrast_to_neighbors.jpg" alt="contrast_to_neighbors" height="250">

### `distance_from_color()`

Sorts pixels based on their proximity to a specified target color in the 3D RGB color space. By default, the target is pure red `(255, 0, 0)`. Pixels closest to the target color will appear first in the sorted output.

<img src="examples/sorted_by_option/distance_from_color.jpg" alt="distance_from_color" height="250">

---

## Positional Sorting

These methods sort pixels based on their position within the slice, not their color. They use luminance as a tie-breaker.

### `distance_center()`

Sorts pixels starting from the middle of the slice and moving outwards towards the edges.

<img src="examples/sorted_by_option/distance_center.jpg" alt="distance_center" height="250">

### `distance_edges()`

Sorts pixels starting from the edges of the slice and moving inwards towards the center.

<img src="examples/sorted_by_option/distance_edges.jpg" alt="distance_edges" height="250">