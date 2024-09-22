import os
import cv2
import numpy as np

# Define the folder containing the license plate images
images_folder = 'D:/Custom License Plate Detection/License Plate Recognition/License Plate Recognition/license_plates_recognition_train'  # Replace with your images folder path
normalized_folder = 'license_plate'  # New folder to save normalized images

# Create the normalized folder if it doesn't exist
os.makedirs(normalized_folder, exist_ok=True)

# Normalize the images
for img_name in os.listdir(images_folder):
    img_path = os.path.join(images_folder, img_name)
    
    # Load the image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Could not load image: {img_path}")
        continue

    # Normalize the image to [0, 1]
    img_normalized = img / 255.0  # Scale pixel values

    # Optionally, you can convert to a specific datatype
    img_normalized = (img_normalized * 255).astype(np.uint8)  # Convert back to uint8 for saving

    # Save the normalized image
    normalized_img_path = os.path.join(normalized_folder, img_name)
    cv2.imwrite(normalized_img_path, img_normalized)

print("Normalization of images completed successfully.")
