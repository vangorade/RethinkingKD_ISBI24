import os
import shutil
import re 

# Define the paths to your source images and labels folders
images_folder = './Task02_Heart/imagesTr/'
labels_folder = './Task02_Heart/labelsTr/'
output_folder = './Task02_Heart/data/data/'
# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in the images folder
image_files = os.listdir(images_folder)

# Define a regular expression pattern to extract the case number
pattern = re.compile(r'la_(\d+)\.nii\.gz')

# Iterate through the image files
for image_file in image_files:
    # Use regular expression to extract the case number
    match = pattern.match(image_file)
    if match:
        case_number = match.group(1)
        case_name = f'case_{case_number.zfill(5)}'
        
        # Define the paths to the source image and label
        image_path = os.path.join(images_folder, image_file)
        label_file = f'la_{case_number}.nii.gz'
        label_path = os.path.join(labels_folder, label_file)
        
        # Create a directory for the case if it doesn't exist
        case_folder = os.path.join(output_folder, case_name)
        os.makedirs(case_folder, exist_ok=True)
        
        # Define the paths for the destination image and label
        dest_image_path = os.path.join(case_folder, 'imaging.nii.gz')
        dest_label_path = os.path.join(case_folder, 'segmentation.nii.gz')
        
        # Copy the source image and label to the case folder
        shutil.copy(image_path, dest_image_path)
        shutil.copy(label_path, dest_label_path)

print("Data folder structure created successfully.")