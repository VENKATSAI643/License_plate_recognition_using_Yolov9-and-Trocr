import pandas as pd

# Load CSV file
csv_file = 'D:/Custom License Plate Detection/License Plate Recognition/License Plate Recognition/Licplatesdetection_train.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Function to convert (ymin, xmin, ymax, xmax) to (center_x, center_y, width, height)
def convert_bbox(ymin, xmin, ymax, xmax):
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return center_x, center_y, width, height

# Define class_id (Assuming a single class, use 0 as default class_id, change if needed)
class_id = 0

# Create a new DataFrame to store the converted values
converted_data = []

for index, row in df.iterrows():
    img_id = row['img_id']
    ymin = row['ymin']
    xmin = row['xmin']
    ymax = row['ymax']
    xmax = row['xmax']
    
    # Convert the bounding box coordinates
    center_x, center_y, width, height = convert_bbox(ymin, xmin, ymax, xmax)
    
    # Append the data as a new row: [img_id, class_id, center_x, center_y, width, height]
    converted_data.append([img_id, class_id, center_x, center_y, width, height])

# Create a new DataFrame with the converted data
columns = ['img_id', 'class_id', 'center_x', 'center_y', 'width', 'height']
converted_df = pd.DataFrame(converted_data, columns=columns)

# Save the new CSV file
output_csv = 'converted_annotations.csv'  # Path to save the new CSV
converted_df.to_csv(output_csv, index=False)

print(f"Converted CSV file saved as {output_csv}")
