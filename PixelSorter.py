import numpy as np
import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from perlin_numpy import generate_fractal_noise_2d
from Enums import SortDirection


class SortBy:
    """Class with static methods for sorting pixels by different criteria."""

    @classmethod
    def list_static_methods(cls):
        """Return all static method names in this class, excluding this one."""
        return [
            name for name, obj in cls.__dict__.items()
            if isinstance(obj, staticmethod) and name != "list_static_methods"
        ]

    def _get_luminance(self, rgb_slice: np.ndarray) -> np.ndarray:
        """Helper to calculate luminance for a slice of RGB pixels."""
        if rgb_slice.ndim == 1: # Handle single pixel case
            return 0.2126 * rgb_slice[0] + 0.7152 * rgb_slice[1] + 0.0722 * rgb_slice[2]
        return 0.2126 * rgb_slice[:, 0] + 0.7152 * rgb_slice[:, 1] + 0.0722 * rgb_slice[:, 2]
    
    @staticmethod
    def hue() -> callable:
        """Sort by hue, then saturation, then brightness (HSV)."""
        return lambda rgb, hsv, lab, idx: (hsv[:, 0], hsv[:, 1], hsv[:, 2])
    
    @staticmethod
    def saturation() -> callable:
        """Sort by saturation, then brightness, then hue (HSV)."""
        return lambda rgb, hsv, lab, idx: (hsv[:, 1], hsv[:, 2], hsv[:, 0])
    
    @staticmethod
    def brightness() -> callable:
        """Sort by brightness (Value from HSV), then saturation, then hue."""
        return lambda rgb, hsv, lab, idx: (hsv[:, 2], hsv[:, 1], hsv[:, 0])
    
    @staticmethod
    def lightness() -> callable:
        """Sort by lightness (L* from Lab color space), which is perceptually uniform."""
        return lambda rgb, hsv, lab, idx: (lab[:, 0],)

    @staticmethod
    def luminance() -> callable:
        """Sort by luminance (calculated perceived brightness from RGB)."""
        return lambda rgb, hsv, lab, idx: (SortBy()._get_luminance(rgb),)

class Image():
    """Class for image processing tasks."""

    @staticmethod
    def load_image(image_path, mode=cv2.IMREAD_COLOR) -> np.ndarray:
        """Load an image from a file path."""
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        return image

    @staticmethod
    def save_image(image: np.ndarray, save_path: str) -> None:
        """Save an image to a file path."""
        cv2.imwrite(save_path, image)

class PixelSorter():
    """Class for sorting pixels in an image based on various criteria."""

    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)


    def _add_perlin_to_image(self, image: np.ndarray, scale: float = 150.0, octaves: int = 4, intensity: float = 0.05) -> np.ndarray:
        """
        Generate a smoothed random noise map and add it to the image.

        Parameters:
            image: input BGR uint8 image
            scale: controls smoothing kernel size (larger -> smoother/low-frequency noise)
            octaves: kept for signature compatibility (unused)
            intensity: how strongly to apply the noise (0.0 - 1.0)

        Returns:
            noisy_image (uint8): image with random smooth perturbation applied
        """
        height, width = image.shape[:2]
        raw_noise = np.random.normal(loc=0.0, scale=1.0, size=(height, width)).astype(np.float32)

        k = max(1, int(round(scale / 50.0)))
        if k % 2 == 0:
            k += 1
        if k < 3:
            k = 3

        blurred_noise = cv2.GaussianBlur(raw_noise, (k, k), 0)

        noise_mean = float(np.mean(blurred_noise))
        noise_std = float(np.std(blurred_noise)) if float(np.std(blurred_noise)) > 0 else 1.0

        normalized = (blurred_noise - noise_mean) / noise_std
        perturbation = normalized * (float(intensity) * 255.0)


        perturbation_3ch = perturbation[:, :, np.newaxis]  # (H,W,1)
        noisy_image_float = image.astype(np.float32) + perturbation_3ch
        noisy_image = np.clip(noisy_image_float, 0, 255).astype(np.uint8)


        return noisy_image


    def sort_pixels(self, sort_by: callable, direction: SortDirection, mask: np.ndarray = None, use_perlin: bool = False) -> np.ndarray:
        """
        Sorts pixels in an image based on specified criteria and an optional grayscale mask.
        If a grayscale mask is provided, the sorted image is blended with the original
        image according to the mask's intensity, creating a "melting" effect.

        Args:
            sort_by (callable): A function that returns sort keys for a slice of pixels.
            direction (SortDirection): The direction (row/column) and order of sorting.
            mask (np.ndarray, optional): A grayscale mask (0-255) defining the blending 
                                          alpha. If None, all pixels are fully sorted. 
                                          Defaults to None.
            use_perlin (bool, optional): Whether to add Perlin noise before sorting for some randomness. Defaults to False.

        Returns:
            np.ndarray: The image with sorted pixels and melting effect.
        """
        image_for_sorting = self.image
        hsv_for_sorting = self.hsv_image
        lab_for_sorting = self.lab_image

        if use_perlin:
            print("Applying Perlin noise...")
            image_for_sorting = self._add_perlin_to_image(self.image)
            # We must update the color space representations based on the noisy image
            hsv_for_sorting = cv2.cvtColor(image_for_sorting, cv2.COLOR_BGR2HSV)
            lab_for_sorting = cv2.cvtColor(image_for_sorting, cv2.COLOR_BGR2Lab)

        original_image = self.image.copy()
        temp_sorted_image = image_for_sorting.copy() # This will hold the fully sorted pixels

        final_mask = mask
        if final_mask is None:
            final_mask = np.full((self.height, self.width), 255, dtype=np.uint8)

        alpha = final_mask.astype(np.float32) / 255.0
        alpha_3_channel = np.stack([alpha, alpha, alpha], axis=-1)
        
        if direction in (SortDirection.ROW_LEFT_TO_RIGHT, SortDirection.ROW_RIGHT_TO_LEFT):
            for y in range(self.height):
                sort_indices_row = np.where(final_mask[y, :] > 0)[0]
                if len(sort_indices_row) < 2: continue

                row_rgb = image_for_sorting[y, sort_indices_row, :]
                row_hsv = hsv_for_sorting[y, sort_indices_row, :]
                row_lab = lab_for_sorting[y, sort_indices_row, :]
                indices_in_slice = np.arange(len(sort_indices_row))

                sort_keys = sort_by(row_rgb, row_hsv, row_lab, indices_in_slice)
                sorted_order = np.lexsort(sort_keys[::-1])

                if direction == SortDirection.ROW_RIGHT_TO_LEFT:
                    sorted_order = sorted_order[::-1]
                
                temp_sorted_image[y, sort_indices_row, :] = row_rgb[sorted_order]
        
        elif direction in (SortDirection.COLUMN_TOP_TO_BOTTOM, SortDirection.COLUMN_BOTTOM_TO_TOP):
            for x in range(self.width):
                sort_indices_col = np.where(final_mask[:, x] > 0)[0]
                if len(sort_indices_col) < 2: continue

                col_rgb = image_for_sorting[sort_indices_col, x, :]
                col_hsv = hsv_for_sorting[sort_indices_col, x, :]
                col_lab = lab_for_sorting[sort_indices_col, x, :]
                indices_in_slice = np.arange(len(sort_indices_col))

                sort_keys = sort_by(col_rgb, col_hsv, col_lab, indices_in_slice)
                sorted_order = np.lexsort(sort_keys[::-1])

                if direction == SortDirection.COLUMN_BOTTOM_TO_TOP:
                    sorted_order = sorted_order[::-1]

                temp_sorted_image[sort_indices_col, x, :] = col_rgb[sorted_order]
        
        final_image = (alpha_3_channel * temp_sorted_image + (1 - alpha_3_channel) * original_image).astype(np.uint8)
        
        return final_image