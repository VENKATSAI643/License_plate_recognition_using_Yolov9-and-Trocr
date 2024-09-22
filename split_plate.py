import os
import shutil
import pandas as pd

# Paths
data_dir = 'D:/Custom License Plate Detection/Plate_recognition/license_plates_recognition_train'
txt_file = 'D:/Custom License Plate Detection/Plate_recognition/license_plate_metadata.txt'
train_dir = 'D:/Custom License Plate Detection/custom_train_plate/train'
eval_dir = 'D:/Custom License Plate Detection/custom_train_plate/eval'

# Create train and eval directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Read the metadata from the txt file (use space delimiter)
df = pd.read_csv(txt_file, sep=' ', header=None, names=["file_name", "text"])

# Check the contents of the DataFrame
print(df.head())  # Print the first few rows to check the structure

# Check the columns
print(df.columns)  # Print the columns to verify names

# Ensure the DataFrame is sorted by index (in case it's not)
df = df.reset_index(drop=True)

# Select images for training (first 700)
train_df = df.iloc[:700]

# Select images for evaluation (from 701 to 900)
eval_df = df.iloc[700:900]

# Copy images to the train folder
for index, row in train_df.iterrows():
    img_src = os.path.join(data_dir, row['file_name'])  # Ensure only the file name is used
    img_dst = os.path.join(train_dir, row['file_name'])
    
    # Check if the file exists before copying
    if os.path.exists(img_src):
        shutil.copy(img_src, img_dst)
    else:
        print(f"File not found: {img_src}")

# Copy images to the eval folder
for index, row in eval_df.iterrows():
    img_src = os.path.join(data_dir, row['file_name'])  # Ensure only the file name is used
    img_dst = os.path.join(eval_dir, row['file_name'])
    
    # Check if the file exists before copying
    if os.path.exists(img_src):
        shutil.copy(img_src, img_dst)
    else:
        print(f"File not found: {img_src}")

# Save train metadata with a new name
train_df.to_csv(os.path.join(train_dir, 'train_metadata.txt'), sep=' ', index=False, header=False)

# Save eval metadata with a new name
eval_df.to_csv(os.path.join(eval_dir, 'eval_metadata.txt'), sep=' ', index=False, header=False)

print("Data has been split and copied to train and eval folders.")

