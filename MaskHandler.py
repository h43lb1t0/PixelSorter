import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from Enums import WhatToSort

class YoloSegmentation():
    """Class for performing YOLO segmentation on images."""

    def __init__(self, image: np.ndarray):
        """
        Initializes the YoloSegmentation class.

        Args:
            image (np.ndarray): The input image as a NumPy array.
        """
        self.model_path = "yolov8x-seg.pt" #"yolo12l-person-seg-extended.pt" 
        self.model = YOLO(self.model_path)
        self.image = image

    def _segment(self, conf: float) -> Results:
        """
        Perform segmentation on the input image.

        Args:
            conf (float): The confidence threshold for detection.

        Returns:
            Results: The segmentation results from the YOLO model.
        """
        results = self.model(self.image, conf=conf)
        print(f"Detected {len(results[0].boxes)} objects with confidence >= {conf}")
        return results[0]

    def get_mask(self, what_to_sort: WhatToSort, conf: float = 0.4, blur_include: float = 1.0, blur_extend: float = 0.0) -> np.ndarray:
        """
        Creates a mask based on the desired sorting target.
        The mask can be grayscale with soft edges, allowing for a "melting" effect.

        Args:
            what_to_sort (WhatToSort): Enum indicating whether to create a mask for
                                     the background, foreground, or all pixels.
            conf (float, optional): The confidence threshold for object detection. Defaults to 0.4.
            blur_include (float, optional): A value from 0.0 to 1.0. Controls how much of
                                            the detected object's core is included before
                                            blurring. 1.0 includes the entire object, while
                                            a smaller value erodes the mask inward, starting
                                            the blur from a smaller base. Defaults to 1.0.
            blur_extend (float, optional): A value from 0.0 to 1.0. Controls the size of the
                                           Gaussian blur kernel applied to the mask,
                                           effectively extending the blurred area outwards
                                           and softening the transition. Defaults to 0.0 (no blur).

        Returns:
            np.ndarray: A mask (0-255 values) representing the area to be sorted.
                        If blur is applied, this will be a grayscale mask. Otherwise,
                        it will be a binary mask (0s and 255s).
        """
        height, width = self.image.shape[:2]

        if what_to_sort == WhatToSort.ALL:
            return np.full((height, width), 255, dtype=np.uint8)

        segmentation = self._segment(conf=conf)
        
        foreground_mask = np.zeros((height, width), dtype=np.uint8)
        if segmentation.masks is not None:
            combined_mask_tensor = torch.sum(segmentation.masks.data, dim=0)
            binary_mask_tensor = (combined_mask_tensor > 0).float()
            mask_np = binary_mask_tensor.cpu().numpy()
            
            resized_mask = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
            foreground_mask = (resized_mask * 255).astype(np.uint8)

            # Step 1: Erode the mask to control how much of the detection is included.
            blur_include_clamped = max(0.0, min(1.0, blur_include))
            if blur_include_clamped < 1.0:
                # A lower blur_include value means more erosion (a larger kernel)
                erosion_factor = 1.0 - blur_include_clamped
                min_dim = min(height, width)
                # This multiplier controls the max erosion amount
                erosion_kernel_dim = int(min_dim * erosion_factor * 0.1) 
                
                if erosion_kernel_dim > 0:
                    # Ensure kernel has an odd dimension
                    if erosion_kernel_dim % 2 == 0:
                        erosion_kernel_dim += 1
                    print(f"Eroding mask with kernel size: {erosion_kernel_dim}")
                    kernel = np.ones((erosion_kernel_dim, erosion_kernel_dim), np.uint8)
                    foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)

            # Step 2: Apply Gaussian blur to extend the mask outwards and soften edges.
            if blur_extend > 0.0:
                blur_intensity = max(0.0, min(1.0, blur_extend))
                max_dim = max(height, width)
                # The blur kernel size determines the outward extension of the blur
                kernel_dim = int(max_dim * blur_intensity * 0.1)
                
                if kernel_dim % 2 == 0:
                    kernel_dim += 1
                if kernel_dim < 3: 
                    kernel_dim = 3

                print(f"Applying Gaussian blur with kernel size: {kernel_dim}")
                foreground_mask = cv2.GaussianBlur(foreground_mask, (kernel_dim, kernel_dim), 0)
        else:
            print("No masks detected.")

        if what_to_sort == WhatToSort.BACKGROUND:
            return 255 - foreground_mask
        else: # WhatToSort.FOREGROUND
            return foreground_mask
