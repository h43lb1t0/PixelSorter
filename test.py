import cv2
import numpy as np
from ultralytics import SAM
from ultralytics.utils.plotting import Colors # Import the Colors class
from PIL import Image
import pathlib

# --- 1. Setup ---
# Define a sample image URL and the desired output path
image_dir = r"E:\Bilder\pixel_input"
output_dir = r"E:\Bilder\pixel_output"

# Find all images in the specified directory

from itertools import chain
images = list(chain(
    pathlib.Path(image_dir).glob("*.jpg"),
    pathlib.Path(image_dir).glob("*.jpeg"),
    pathlib.Path(image_dir).glob("*.png")
))

# create output directory if it doesn't exist
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


# --- 2. Load Model ---
# Load the Segment Anything Model (SAM).
print("Loading SAM model...")
model = SAM('sam2.1_l.pt') # Note: Assuming you have the correct model file
model.info()
print("Model loaded successfully.")

for image_path in images:
    image_path = str(image_path)
    output_path = str(pathlib.Path(output_dir) / (pathlib.Path(image_path).stem + "-segmented.jpg"))

    print(f"\nProcessing image: {image_path}")

    image_path = str(image_path)

    # --- 3. Load Image ---
    # Load the image from the local file system
    try:
        image = Image.open(image_path).convert("RGB")
        print("Image loaded successfully.")
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    # --- 4. Run Segmentation ---
    # Run the SAM model on the loaded image to find all possible segments.
    print("Running segmentation...")
    results = model(image, device=0)
    print("Segmentation complete.")

    # Start with the original image in BGR format for OpenCV
    final_image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Check if any masks were detected
    if results[0].masks:
        masks_tensor = results[0].masks.data
        
        # === NEW: 6. Create and Save Combined Binary Mask ===
        print("Creating a combined binary mask...")

        # Combine all masks into a single binary mask using a logical OR operation.
        # The .any(dim=0) operation efficiently collapses the stack of masks.
        # If any mask has a 1 at a pixel location, the resulting mask will have a 1.
        combined_mask_tensor = masks_tensor.any(dim=0)

        # Convert the boolean tensor to a NumPy array of type uint8 (0s and 255s)
        # This is the standard format for binary images.
        binary_mask_np = combined_mask_tensor.cpu().numpy().astype(np.uint8) * 255
        
        # Create a PIL Image from the NumPy array
        binary_mask_image = Image.fromarray(binary_mask_np)

        # Define the output path for the binary mask image
        binary_mask_output_path = str(pathlib.Path(output_dir) / (pathlib.Path(image_path).stem + "-binary-mask.jpg"))

        # Save the binary mask image
        binary_mask_image.save(binary_mask_output_path)

        print(f"💾 Combined binary mask has been saved to '{binary_mask_output_path}'")

        # === Part 5.1: Apply colored, semi-transparent masks ===
        # Create a colors object to generate a unique color for each mask
        colors = Colors()

        print(f"Applying colors for {len(masks_tensor)} masks...")
        for i, mask in enumerate(masks_tensor):
            # Get a unique color for this mask instance
            color = colors(i, bgr=True)
            
            # Get the binary mask as a boolean NumPy array
            mask_np_bool = mask.cpu().numpy().astype(bool)

            # Apply the color with transparency
            # Where the mask is True, blend the original color with the new color
            final_image_bgr[mask_np_bool] = cv2.addWeighted(
                src1=final_image_bgr[mask_np_bool],
                alpha=0.4,  # Opacity of the original image region (40%)
                src2=np.full_like(final_image_bgr[mask_np_bool], color),
                beta=0.6,   # Opacity of the new color (60%)
                gamma=0
            )

        # === Part 5.2: Draw the black outlines on top of the colored masks ===
        print(f"Drawing outlines for {len(masks_tensor)} masks...")
        for mask in masks_tensor:
            # Get the mask as a uint8 array (required for findContours)
            mask_np_uint8 = mask.cpu().numpy().astype(np.uint8)

            # Find contours for this single mask
            contours, _ = cv2.findContours(mask_np_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw the found contours on the image
            cv2.drawContours(
                image=final_image_bgr,
                contours=contours,
                contourIdx=-1,
                color=(0, 0, 0),  # Black color in BGR
                thickness=3       # Line thickness
            )

    # Convert the final BGR image back to an RGB PIL Image for saving.
    plotted_image_rgb = Image.fromarray(cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB))

    # Save the final image.
    plotted_image_rgb.save(output_path)

    print(f"✅ Segmented image with colored masks and black outlines has been saved to '{output_path}'")