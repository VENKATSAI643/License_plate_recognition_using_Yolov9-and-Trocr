import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

# Load the converted CSV file
csv_file = 'converted_annotations.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Define the images folder
images_folder = 'D:/Custom License Plate Detection/License Plate Recognition/License Plate Recognition/license_plates_detection_train'  # Replace with your images folder path

# Function to normalize the bounding box values
def normalize_bbox(center_x, center_y, width, height, img_width, img_height):
    norm_center_x = center_x / img_width
    norm_center_y = center_y / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return norm_center_x, norm_center_y, norm_width, norm_height

# Function to denormalize the bounding box values for visualization
def denormalize_bbox(norm_center_x, norm_center_y, norm_width, norm_height, img_width, img_height):
    center_x = norm_center_x * img_width
    center_y = norm_center_y * img_height
    width = norm_width * img_width
    height = norm_height * img_height
    xmin = int(center_x - width / 2)
    ymin = int(center_y - height / 2)
    xmax = int(center_x + width / 2)
    ymax = int(center_y + height / 2)
    return xmin, ymin, xmax, ymax

# Function to visualize the bounding boxes on the images
def visualize_bboxes(image_path, bboxes, img_width, img_height):
    # Load the image
    img = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if img is None:
        print(f"Image not found: {image_path}")
        return
    
    # Draw each bounding box on the image
    for bbox in bboxes:
        class_id, norm_center_x, norm_center_y, norm_width, norm_height = bbox
        # Convert normalized coordinates back to pixel coordinates
        xmin, ymin, xmax, ymax = denormalize_bbox(norm_center_x, norm_center_y, norm_width, norm_height, img_width, img_height)
        # Draw a rectangle around the detected object
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Optionally, display the class id on the top-left corner
        cv2.putText(img, f"Class {class_id}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert from BGR (OpenCV format) to RGB (Matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image with bounding boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Group the CSV by img_id so we can process all bounding boxes for a single image
grouped = df.groupby('img_id')

# List to store normalized data
normalized_data = []

# Loop through each image in the CSV
for img_id, group in grouped:
    # Get the path of the image
    image_path = os.path.join(images_folder, img_id)
    
    # Load the image to get its dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        continue
    
    img_height, img_width, _ = img.shape
    
    # Get the bounding box values for this image and normalize them
    bboxes = []
    for _, row in group.iterrows():
        class_id = row['class_id']
        center_x = row['center_x']
        center_y = row['center_y']
        width = row['width']
        height = row['height']
        
        # Normalize the bounding box values
        norm_center_x, norm_center_y, norm_width, norm_height = normalize_bbox(center_x, center_y, width, height, img_width, img_height)
        
        # Append normalized values
        normalized_data.append([img_id, class_id, norm_center_x, norm_center_y, norm_width, norm_height])
        
        # Store normalized bounding boxes for visualization
        bboxes.append([class_id, norm_center_x, norm_center_y, norm_width, norm_height])
    
    # Visualize the image with normalized bounding boxes (after converting back to pixel coordinates)
    visualize_bboxes(image_path, bboxes, img_width, img_height)

# Create a DataFrame for normalized data and save to a new CSV file
normalized_columns = ['img_id', 'class_id', 'norm_center_x', 'norm_center_y', 'norm_width', 'norm_height']
normalized_df = pd.DataFrame(normalized_data, columns=normalized_columns)
normalized_df.to_csv('normalized_annotations.csv', index=False)

print("Normalized CSV file saved as 'normalized_annotations.csv'")
