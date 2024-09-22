import os
import shutil
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# Load the original CSV file
csv_file = 'converted_annotations.csv'  # Replace with your original CSV file path
df = pd.read_csv(csv_file)

# Define the images folder
images_folder = 'D:/Custom License Plate Detection/License Plate Recognition/License Plate Recognition/license_plates_detection_train'  # Replace with your images folder path

# Define the target directory for the dataset
output_dir = 'custom_dataset'  # The base directory for train and validation folders
train_folder = os.path.join(output_dir, 'train')
val_folder = os.path.join(output_dir, 'validation')

# Create the directory structure if it doesn't exist
for folder in [train_folder, val_folder]:
    os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'labels'), exist_ok=True)

# Function to normalize the bounding box values
def normalize_bbox(center_x, center_y, width, height, img_width, img_height):
    norm_center_x = center_x / img_width
    norm_center_y = center_y / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    return norm_center_x, norm_center_y, norm_width, norm_height

# Function to save labels to a text file in the required format
def save_label_file(image_name, bboxes, folder):
    label_path = os.path.join(folder, 'labels', f'{image_name}.txt')
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            class_id, norm_center_x, norm_center_y, norm_width, norm_height = bbox
            # Write the normalized bbox values to the file
            f.write(f'{class_id} {norm_center_x} {norm_center_y} {norm_width} {norm_height}\n')

# Function to copy images to the correct folder
def copy_image(image_name, src_folder, dst_folder):
    src_image_path = os.path.join(src_folder, image_name)
    dst_image_path = os.path.join(dst_folder, 'images', image_name)
    shutil.copyfile(src_image_path, dst_image_path)

# Helper function to process data
def process_data(dataframe, src_images_folder, target_folder):
    grouped = dataframe.groupby('img_id')

    for img_id, group in grouped:
        # Load the image to get its dimensions
        image_path = os.path.join(src_images_folder, img_id)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image not found: {image_path}")
            continue

        img_height, img_width, _ = img.shape

        # Get the bounding box values for the image and normalize them
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
            bboxes.append([class_id, norm_center_x, norm_center_y, norm_width, norm_height])

        # Copy the image to the appropriate folder (train or validation)
        copy_image(img_id, src_images_folder, target_folder)
        # Save the labels in the .txt file with the image name
        save_label_file(img_id.split('.')[0], bboxes, target_folder)

# Split data into train and validation (80% train, 20% validation)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Process the train data
process_data(train_df, images_folder, train_folder)

# Process the validation data
process_data(val_df, images_folder, val_folder)

print("Dataset directory structure created successfully.")
