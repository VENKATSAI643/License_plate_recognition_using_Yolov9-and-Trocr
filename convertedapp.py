import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

# Load the converted CSV file
csv_file = 'converted_annotations.csv'  # Replace with your converted CSV file path
df = pd.read_csv(csv_file)

# Define the images folder
images_folder = 'D:/Custom License Plate Detection/License Plate Recognition/License Plate Recognition/license_plates_detection_train'  # Replace with your images folder path

# Function to visualize bounding boxes on the images
def visualize_bboxes(image_path, bboxes):
    # Load the image
    img = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if img is None:
        print(f"Image not found: {image_path}")
        return
    
    # Draw each bounding box on the image
    for bbox in bboxes:
        class_id, center_x, center_y, width, height = bbox
        xmin = int(center_x - (width / 2))
        ymin = int(center_y - (height / 2))
        xmax = int(center_x + (width / 2))
        ymax = int(center_y + (height / 2))
        
        # Draw a rectangle around the detected object
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Put class_id text on the bounding box
        cv2.putText(img, f'Class {class_id}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Convert from BGR (OpenCV format) to RGB (Matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image with bounding boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

# Group the CSV by img_id so we can process all bounding boxes for a single image
grouped = df.groupby('img_id')

# Loop through each image in the CSV
for img_id, group in grouped:
    # Get the path of the image
    image_path = os.path.join(images_folder, img_id)
    
    # Get the bounding box values for this image (class_id, center_x, center_y, width, height)
    bboxes = group[['class_id', 'center_x', 'center_y', 'width', 'height']].values
    
    # Visualize the image with bounding boxes
    visualize_bboxes(image_path, bboxes)
