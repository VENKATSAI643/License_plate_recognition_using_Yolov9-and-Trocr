
# License Plate Recognition

recognizing and extracting text from license plates, built using YOLOv9 for object detection and Microsoft’s TrOCR (Text Recognition) model. The system was collaboratively developed by me and my friend to handle the entire process from data preprocessing to model training and evaluation.

## Project Overview

The goal of this project is to recognize and extract text from license plates. The pipeline involves:
- Preprocessing the dataset, including normalizing the image sizes.
- Converting the dataset format to work with YOLOv9 for object detection.
- Formatting the dataset for training with Microsoft's TrOCR model for text recognition.
- Custom training of the TrOCR model on the license plate dataset.

## Dataset Preprocessing

### Step 1: Converting CSV Format for YOLOv9
The original dataset provided the bounding box coordinates in the format:
- `ymin`, `ymax`, `xmin`, `xmax`

These values were converted to the YOLOv9 format, which requires:
- `class_id`, `center_x`, `center_y`, `width`, `height`

This conversion was done by:
1. Calculating the center coordinates of the bounding box (`center_x`, `center_y`).
2. Calculating the width and height of the bounding box.
3. Normalizing these values by the image dimensions.

### Step 2: Image Normalization
To ensure consistency in training, all images were resized and normalized to have a uniform size. This step is crucial for improving model performance and reducing training inconsistencies.

### Step 3: Formatting for TrOCR
For text recognition, the dataset was converted to a format suitable for Microsoft's TrOCR model. This involved:
1. Restructuring the image filenames and the corresponding text data.
2. Ensuring that the dataset follows the required format for custom training with TrOCR.

## Training Process

### YOLOv9 for Object Detection
YOLOv9 was trained to detect license plates in the images. After preprocessing the bounding box data, the model was custom trained using the converted dataset.

### TrOCR for Text Recognition
Microsoft’s TrOCR model was customized to recognize the text on the detected license plates. This involved:
- Formatting the dataset as required by TrOCR.
- Fine-tuning the model on the specific dataset of license plates for improved accuracy.

## Results

After training both models, the system is capable of:
1. Detecting license plates in images using YOLOv9.
2. Extracting and recognizing the text from the license plates using the fine-tuned TrOCR model.
