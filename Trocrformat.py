import pandas as pd

# Paths
csv_file = 'D:/Custom License Plate Detection/License Plate Recognition/License Plate Recognition/Licplatesrecognition_train.csv'
txt_file = 'metadata.txt'

# Read the CSV file
df = pd.read_csv(csv_file)

# Extract numeric part from img_id for sorting
df['img_number'] = df['img_id'].str.extract('(\d+)').astype(int)

# Sort the DataFrame by the extracted image numbers
df = df.sort_values(by='img_number')

# Write to the TXT file
with open(txt_file, 'w') as f:
    for index, row in df.iterrows():
        f.write(f"{row['img_id']} {row['text']}\n")

print("Sorted TXT file has been created.")
